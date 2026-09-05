"""MechRL baseline for circuit discovery.

Method: MechRL — uses reinforcement learning to discover circuits.

RL formulation (on-the-fly REINFORCE):
  - State:   current binary mask over heads (which are "in" the circuit)
  - Action:  toggle one head in or out of the current circuit mask
  - Reward:  improvement in task metric per step, penalized by circuit size
  - Policy:  small MLP (state → action logits), trained via REINFORCE

The policy is trained from scratch on the target task using N episodes of
trajectory collection followed by policy-gradient updates. Greedy decoding
then produces the final circuit.

This keeps MechRL fully self-contained (no pre-training required) while
capturing the RL-based exploration approach distinct from gradient descent
(Subnetwork Probing) or Bayesian belief updates (ACD).
"""
from __future__ import annotations

import time
import random
from typing import NamedTuple

import torch
import torch.nn as nn

from ..tools import adapter
from . import MethodResult

# Hyperparameters
_DEFAULT_N_EPISODES       = 60    # RL training episodes
_DEFAULT_EPISODE_LEN      = 15    # max steps per episode
_DEFAULT_LR_POLICY        = 5e-3  # policy network learning rate
_DEFAULT_GAMMA            = 0.95  # discount factor
_DEFAULT_SIZE_PENALTY     = 0.02  # penalty per head in circuit (encourages sparsity)
_DEFAULT_CIRCUIT_FREQ_THR = 0.4   # include head if selected in > this fraction of eval rollouts
_DEFAULT_EVAL_EPISODES    = 20    # greedy rollouts to determine final circuit


class _Trajectory(NamedTuple):
    actions:  list[int]   # indices into flat head list
    log_probs: list[float]
    rewards:  list[float]


class _PolicyNet(nn.Module):
    """Tiny 2-layer MLP: state (binary mask) → logits over actions (toggle each head)."""
    def __init__(self, n_heads_total: int):
        super().__init__()
        hidden = max(32, n_heads_total // 2)
        self.net = nn.Sequential(
            nn.Linear(n_heads_total, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_heads_total),
        )

    def forward(self, mask_state: torch.Tensor) -> torch.Tensor:
        return self.net(mask_state.float())


def _flat_idx_to_lh(flat_idx: int, layers: list[int], n_heads: int) -> tuple[int, int]:
    l_pos = flat_idx // n_heads
    h     = flat_idx % n_heads
    return layers[l_pos], h


def _run_episode(
    policy: _PolicyNet,
    handle: adapter.ModelHandle,
    layers: list[int],
    n_heads: int,
    io_id: int,
    s_id: int,
    ci: dict,
    xi: dict,
    clean_m: float,
    corr_m: float,
    episode_len: int,
    size_penalty: float,
    greedy: bool = False,
) -> _Trajectory:
    """Collect one trajectory under the current policy."""
    n_flat = len(layers) * n_heads
    mask_state = torch.zeros(n_flat, device=handle.device)  # all heads off initially
    denom = clean_m - corr_m

    actions_taken: list[int]   = []
    log_probs_taken: list[float] = []
    rewards_taken: list[float] = []

    prev_metric = corr_m

    for _ in range(episode_len):
        # Policy forward
        with torch.no_grad():
            logits = policy(mask_state.unsqueeze(0))[0]
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            action = int(probs.argmax().item())
        else:
            action = int(torch.multinomial(probs, 1).item())

        log_p = float(torch.log(probs[action] + 1e-12).item())

        # Toggle action: flip head in/out
        mask_state = mask_state.clone()
        mask_state[action] = 1.0 - mask_state[action]

        # Evaluate current circuit
        circuit_heads = [
            _flat_idx_to_lh(i, layers, n_heads)
            for i in range(n_flat) if mask_state[i] > 0.5
        ]
        all_model_heads = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
        complement = [lh for lh in all_model_heads if lh not in set(circuit_heads)]

        if circuit_heads and abs(denom) > 1e-8:
            metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())
            curr_metric = float(adapter.ablate_head_set(
                handle, complement,
                ci["input_ids"], ci["attention_mask"],
                metric_fn,
            ))
        else:
            curr_metric = corr_m

        # Reward: metric improvement – size penalty
        delta   = curr_metric - prev_metric
        n_in    = int(mask_state.sum().item())
        reward  = delta - size_penalty * n_in
        prev_metric = curr_metric

        actions_taken.append(action)
        log_probs_taken.append(log_p)
        rewards_taken.append(reward)

    return _Trajectory(actions=actions_taken, log_probs=log_probs_taken, rewards=rewards_taken)


def _compute_returns(rewards: list[float], gamma: float) -> list[float]:
    """Discounted returns G_t = r_t + γ*r_{t+1} + ..."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def run_mechrl(
    handle: adapter.ModelHandle,
    task: dict,
    layer_range: range | None = None,
    n_heads: int | None = None,
    n_episodes: int = _DEFAULT_N_EPISODES,
    episode_len: int = _DEFAULT_EPISODE_LEN,
    lr_policy: float = _DEFAULT_LR_POLICY,
    gamma: float = _DEFAULT_GAMMA,
    size_penalty: float = _DEFAULT_SIZE_PENALTY,
    circuit_freq_thr: float = _DEFAULT_CIRCUIT_FREQ_THR,
    eval_episodes: int = _DEFAULT_EVAL_EPISODES,
    seed: int = 0,
    **kwargs,
) -> MethodResult:
    """Run MechRL circuit discovery on `task`.

    Args:
        handle:           ModelHandle for the target model.
        task:             Task dict (same schema as behaviors.py / task_suites).
        layer_range:      Layers to search over (default: all layers).
        n_heads:          Number of attention heads (default: from task["n_heads"]).
        n_episodes:       Number of RL training episodes.
        episode_len:      Max steps per episode.
        lr_policy:        Policy network learning rate.
        gamma:            Discount factor for returns.
        size_penalty:     Reward penalty per head in circuit (sparsity pressure).
        circuit_freq_thr: Head selected in > this fraction of eval rollouts → in circuit.
        eval_episodes:    Number of greedy rollouts for final circuit determination.
        seed:             RNG seed for reproducibility.

    Returns:
        MethodResult with discovered circuit, metric recovery, runtime, etc.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)

    n_heads     = n_heads or task["n_heads"]
    layer_range = layer_range or range(handle.n_layers)
    layers      = list(layer_range)
    n_flat      = len(layers) * n_heads

    cp, xp   = task["clean_prompt"], task["corrupted_prompt"]
    io_t, s_t = task["io_token"], task["s_token"]

    # [-1] not [0]: cross-tokenizer safety (see behaviors.py _make_task)
    io_id = handle.tokenizer.encode(" " + io_t.strip())[-1]
    s_id  = handle.tokenizer.encode(" " + s_t.strip())[-1]

    ci = handle.tokenizer([cp], return_tensors="pt").to(handle.device)
    xi = handle.tokenizer([xp], return_tensors="pt").to(handle.device)

    metric_fn = lambda logits: float((logits[0, -1, io_id] - logits[0, -1, s_id]).item())
    with adapter.MODEL_LOCK, torch.no_grad():
        clean_m = metric_fn(handle.model(**ci).logits)
        corr_m  = metric_fn(handle.model(**xi).logits)

    # Initialize policy network
    policy    = _PolicyNet(n_flat).to(handle.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_policy)

    total_fwd_passes = 2  # for clean_m and corr_m

    # ---- Training loop ----
    episode_returns: list[float] = []
    for ep in range(n_episodes):
        traj = _run_episode(
            policy, handle, layers, n_heads,
            io_id, s_id, ci, xi, clean_m, corr_m,
            episode_len, size_penalty, greedy=False,
        )
        total_fwd_passes += len(traj.actions)  # one ablation eval per step

        # REINFORCE update
        returns  = _compute_returns(traj.rewards, gamma)
        baseline = sum(returns) / len(returns)  # simple mean baseline
        policy_loss = torch.tensor(0.0, device=handle.device, requires_grad=True)
        for log_p, G in zip(traj.log_probs, returns):
            policy_loss = policy_loss + (-log_p * (G - baseline))
        policy_loss = policy_loss / len(traj.actions)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        episode_returns.append(sum(traj.rewards))

    # ---- Evaluation: greedy rollouts to determine final circuit ----
    head_counts = [0] * n_flat
    for _ in range(eval_episodes):
        traj = _run_episode(
            policy, handle, layers, n_heads,
            io_id, s_id, ci, xi, clean_m, corr_m,
            episode_len, size_penalty, greedy=True,
        )
        total_fwd_passes += len(traj.actions)
        # Count which heads were toggled ON at end (track mask state manually)
        mask = [0.0] * n_flat
        for a in traj.actions:
            mask[a] = 1.0 - mask[a]
        for i, v in enumerate(mask):
            if v > 0.5:
                head_counts[i] += 1

    # Heads selected in > circuit_freq_thr fraction of eval rollouts → in circuit
    circuit = sorted(
        [_flat_idx_to_lh(i, layers, n_heads)
         for i, cnt in enumerate(head_counts)
         if cnt / eval_episodes > circuit_freq_thr],
        key=lambda lh: -head_counts[
            layers.index(lh[0]) * n_heads + lh[1]
        ],
    )

    # Measure metric recovery
    denom = clean_m - corr_m
    all_model_heads = [(l, h) for l in range(handle.n_layers) for h in range(n_heads)]
    complement = [lh for lh in all_model_heads if lh not in set(circuit)]

    if circuit and abs(denom) > 1e-8:
        ablated_m = float(adapter.ablate_head_set(
            handle, complement,
            ci["input_ids"], ci["attention_mask"],
            metric_fn,
        ))
        metric_recovery = (ablated_m - corr_m) / denom
        total_fwd_passes += 1
    else:
        metric_recovery = 0.0

    runtime = time.time() - t0

    return MethodResult(
        method="mechrl",
        circuit=circuit,
        circuit_score=max(head_counts) / eval_episodes if head_counts else 0.0,
        metric_recovery=metric_recovery,
        tool_calls=total_fwd_passes,
        runtime_s=runtime,
        metadata={
            "n_episodes": n_episodes, "episode_len": episode_len,
            "lr_policy": lr_policy, "gamma": gamma, "size_penalty": size_penalty,
            "circuit_freq_thr": circuit_freq_thr, "eval_episodes": eval_episodes,
            "clean_m": clean_m, "corr_m": corr_m,
            "mean_episode_return": (sum(episode_returns) / len(episode_returns)
                                    if episode_returns else 0.0),
            "head_selection_freq": {
                str(_flat_idx_to_lh(i, layers, n_heads)): round(cnt / eval_episodes, 3)
                for i, cnt in enumerate(head_counts) if cnt > 0
            },
        },
    )

