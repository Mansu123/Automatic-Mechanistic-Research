import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2Config, BatchEncoding

import run_colab as runner
from automechinterp import behaviors
from automechinterp.tools import adapter, colab_causal
from automechinterp.eval.causal_measurements import score_contrast, continuation_log_probability
from automechinterp.techniques import _common as C, probing, superposition, safety, weight_space
from automechinterp.agents.base import Agent, ToolCallBudget
from automechinterp.llm_backends import HeuristicBackend
from test_evaluation_science import small_configs


class Tokenizer:
    pad_token_id = 0
    pad_token = "pad"
    eos_token = "end"
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [3+ord(c)%50 for c in text]
    def decode(self, ids):
        return " ".join("t"+str(i) for i in ids)
    def __call__(self, texts, return_tensors="pt", padding=False, **kwargs):
        if isinstance(texts, str): texts=[texts]
        rows=[self.encode(t) for t in texts]; n=max(map(len, rows))
        return BatchEncoding({"input_ids":torch.tensor([r+[0]*(n-len(r)) for r in rows]),
            "attention_mask":torch.tensor([[1]*len(r)+[0]*(n-len(r)) for r in rows])})


def fixture(cfg=None, dtype=torch.float32, device="cpu"):
    cfg=cfg or GPT2Config(vocab_size=64,n_embd=32,n_layer=2,n_head=4,n_positions=512,bos_token_id=1,eos_token_id=2)
    cfg.max_position_embeddings=512
    if hasattr(cfg,"n_positions"): cfg.n_positions=512
    torch.manual_seed(7)
    model=AutoModelForCausalLM.from_config(cfg,attn_implementation="eager").to(device=device,dtype=dtype).eval()
    model.requires_grad_(False);model.config.use_cache=False
    path,layers=adapter._discover_layers(model)
    return adapter.ModelHandle(model,Tokenizer(),cfg.model_type,device,path,layers)


class ColabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_exact_22_by_203_manifest_and_all_25_angles(self):
        config=runner.load_config(runner.REPO/"colab_config.json")
        names=[b.__name__ for b in behaviors.ALL_BEHAVIORS]
        self.assertEqual(names,config["behavior_ids"])
        h=fixture(); counts=Counter()
        for builder in behaviors.ALL_BEHAVIORS:
            task=builder(h,seed=0)
            counts[int(task["category"].split(":")[0].split()[-1])]+=1
            evaluated=task["eval_prompts"]
            self.assertTrue(evaluated, builder.__name__)
            self.assertEqual(len(evaluated),len(set(map(tuple,evaluated))))
            self.assertFalse({r[0] for r in evaluated}&set(task["probe_texts"]))
            self.assertFalse(task["data_audit"]["independent_stress_validated"])
        self.assertEqual(counts,runner.ANGLE_COUNTS)
        self.assertEqual(len(config["models"])*len(names)*len(config["seeds"]),4466)
        self.assertNotIn("EleutherAI/pythia-70m",[m["model_id"] for m in config["models"]])

    def test_batched_full_answers_match_sequential_with_shared_prefix(self):
        h=fixture()
        for a,b in [("x","y"),("red apple","green apple"),("New York","New Jersey")]:
            result=score_contrast(h,"The answer is",a,b)
            pos,_=continuation_log_probability(h,"The answer is",a)
            neg,_=continuation_log_probability(h,"The answer is",b)
            self.assertAlmostEqual(result["margin"],pos-neg,places=4)

    def test_bf16_hooks_probes_sae_and_mlp_paths_on_every_architecture(self):
        device="cuda" if torch.cuda.is_available() else "cpu"
        pos=["nice day "+str(i) for i in range(6)]
        neg=["bad day "+str(i) for i in range(6)]
        for cfg in small_configs():
            with self.subTest(architecture=cfg.model_type,device=device):
                h=fixture(cfg,torch.bfloat16,device)
                self.assertIsNotNone(adapter._find_mlp_out_proj(h.layers[0]))
                vector=torch.randn(h.model.config.hidden_size)
                self.assertTrue(C.top_tokens(h,vector,3))
                for hook in [C.add_direction_hook(vector,1),C.project_out_hook(vector)]:
                    x=torch.randn(2,4,len(vector),device=device,dtype=torch.bfloat16)
                    out=hook(None,None,x)
                    self.assertEqual(out.dtype,x.dtype);self.assertEqual(out.device,x.device)
                result=probing.linear_probe(h,0,pos,neg)
                self.assertIn("cv_accuracy",result)
                result=superposition.train_toy_sae(h,0,pos+neg,expansion=1,steps=2)
                self.assertTrue(np.isfinite(result["FVU"]))
                self.assertEqual(weight_space.parameter_svd(h,"mlp_out",max_layers=1)["matrix"],"mlp_out")
                self.assertTrue(weight_space.parameter_svd(h,"mlp_out",max_layers=1)["reused_immutable_model_measurement"])

    def test_cka_dual_form_is_exact_and_capture_keeps_last_valid_token(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(8,64));y=rng.normal(size=(8,64))
        xc=x-x.mean(0);yc=y-y.mean(0)
        old=np.linalg.norm(xc.T@yc)**2/(np.linalg.norm(xc.T@xc)*np.linalg.norm(yc.T@yc))
        self.assertAlmostEqual(adapter._linear_cka(x,y),old,places=12)
        h=fixture();texts=["hi","a much longer phrase","end"]
        with patch.dict("os.environ",{"AMI_CAPTURE_BATCH_SIZE":"2"}):
            batch=adapter.capture_activations(h,[0,1],texts)
        for i,t in enumerate(texts):
            one=adapter.capture_activations(h,[0,1],[t])
            for layer in (0,1):np.testing.assert_allclose(batch[layer][i],one[layer][0],atol=1e-6)

    def test_multilayer_gradients_remain_connected_and_unequal_prompt_patching_works(self):
        h=fixture()
        g,a=colab_causal.gradients(h,"A short prompt","yes","no",[0,1])
        self.assertGreater(float(g[0].abs().sum()),0)
        # This character tokenizer's first continuation token is the shared space.
        # The final layer at the prompt position cannot affect later tokens: its
        # exact cancellation is a useful negative control, while layer 0 stays connected.
        self.assertAlmostEqual(float(g[1].abs().sum()),0,places=6)
        result=colab_causal.layer_effect(h,0,"Here is the answer","A different longer answer prompt","yes","no")
        self.assertTrue(np.isfinite(result["patched_metric"]))
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in h.model.modules()))

    def test_resume_requires_exact_identity_checksum_and_no_tool_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"local";persist=Path(temp)/"drive"
            ident={"schema":2,"run_id":"new","model":"gpt2","revision":"a"*40,"behavior_id":"ioi_behavior","seed":0}
            payload={"identity":ident,"status":"completed","report_markdown":"evidence",
                "entry":{"execution_status":"completed","raw_hierarchy_result":{"execution_status":"completed","agent_runs":[{"errors":[]}]}}}
            runner.save_checkpoint(root,persist,payload)
            self.assertIsNotNone(runner.read_checkpoint(runner.job_path(root,ident),ident))
            self.assertIsNone(runner.read_checkpoint(runner.job_path(root,ident),{**ident,"run_id":"other"}))
            runner.job_path(root,ident).write_text("truncated")
            self.assertIsNotNone(runner.restore(root,persist,ident))
            payload["entry"]["raw_hierarchy_result"]["agent_runs"][0]["errors"]=["bad_tool"]
            runner.save_checkpoint(root,persist,payload)
            self.assertIsNone(runner.read_checkpoint(runner.job_path(root,ident),ident))

    def test_legacy_files_do_not_skip_any_of_4466_jobs(self):
        config=runner.load_config(runner.REPO/"colab_config.json")
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,"stage_d_gpt2_legacy.json").write_text('{"status":"ok"}')
            result=runner.coverage(config,temp,"new")
            self.assertEqual(result["expected"],4466);self.assertEqual(result["completed"],0)
            self.assertFalse(result["all_22_models_203_behaviors_complete"])

    def test_swallowed_error_string_is_still_a_failed_tool(self):
        agent=Agent(HeuristicBackend(lambda *a:{"action":"example"}),
                    {"example":lambda:"ERROR: RuntimeError: bad dtype"},ToolCallBudget([2],2))
        with contextlib.redirect_stdout(io.StringIO()): agent.run()
        self.assertEqual(agent.telemetry[0]["status"],"error")
        self.assertIn("example",agent._failed_tools)


if __name__=="__main__": unittest.main()
