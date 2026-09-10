import unittest
from unittest.mock import patch

import torch
from transformers import (AutoModelForCausalLM, GPT2Config, GPTNeoXConfig, GPTNeoConfig,
                          OPTConfig, PhiConfig, LlamaConfig, Gemma2Config, Qwen2Config, MistralConfig)

from automechinterp.tools import adapter, verification
from automechinterp.eval.causal_measurements import normalized_effect, score_contrast, ablate_heads, proportion
from automechinterp.agents.base import Agent, ToolCallBudget
from automechinterp.agents.judge import adjudicate
from automechinterp.llm_backends import HeuristicBackend
from automechinterp.stage_a import score_against_ground_truth
from automechinterp.eval.datasets import validate_dataset,build_ioi_calibration_dataset


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [3 + ord(c) % 50 for c in text]


def small_configs():
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128)
    return [
        GPT2Config(vocab_size=64,n_embd=32,n_layer=2,n_head=4,n_positions=128),
        GPTNeoXConfig(**common,rotary_pct=.5),
        GPTNeoConfig(vocab_size=64,hidden_size=32,num_layers=2,num_heads=4,
                     intermediate_size=64,max_position_embeddings=128,attention_types=[[['global','local'],1]]),
        OPTConfig(**common,ffn_dim=64,word_embed_proj_dim=32),
        PhiConfig(**common,partial_rotary_factor=.5),
        LlamaConfig(**common,num_key_value_heads=2,head_dim=8),
        Gemma2Config(**{**common,'hidden_size':36},num_key_value_heads=2,head_dim=8),
        Qwen2Config(**common,num_key_value_heads=2),
        MistralConfig(**common,num_key_value_heads=2,head_dim=8),
    ]


class EvaluationScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_architecture_projection_and_identity_interventions(self):
        for cfg in small_configs():
            with self.subTest(architecture=cfg.model_type):
                torch.manual_seed(5)
                model=AutoModelForCausalLM.from_config(cfg,attn_implementation='eager').eval()
                path,layers=adapter._discover_layers(model)
                handle=adapter.ModelHandle(model,CharacterTokenizer(),cfg.model_type,'cpu',path,layers)
                self.assertEqual(adapter.get_head_dim(handle),8)
                ids=torch.tensor([[2,3,4,5]])
                mask=torch.ones_like(ids)
                metric=lambda x:float(x[0,-1,7]-x[0,-1,8])
                with torch.no_grad(): base=metric(model(input_ids=ids,attention_mask=mask).logits)
                same=adapter.run_with_head_patch(handle,0,2,ids,mask,ids,mask,metric)
                self.assertAlmostEqual(base,same,places=6)
                with ablate_heads(handle,[(0,1),(1,2)]):
                    with torch.no_grad(): altered=metric(model(input_ids=ids,attention_mask=mask).logits)
                self.assertNotAlmostEqual(base,altered,places=8)
                with torch.no_grad(): restored=metric(model(input_ids=ids,attention_mask=mask).logits)
                self.assertAlmostEqual(base,restored,places=6)
                with self.assertRaises(ValueError):
                    adapter.run_with_head_patch(handle,0,0,ids,mask,ids[:,:-1],mask[:,:-1],metric)
                for layer in layers:
                    self.assertFalse(adapter._find_attn_out_proj(layer)._forward_pre_hooks)

    def test_multitoken_candidate_scores_entire_continuation(self):
        cfg=GPT2Config(vocab_size=64,n_embd=32,n_layer=2,n_head=4,n_positions=128)
        model=AutoModelForCausalLM.from_config(cfg).eval()
        path,layers=adapter._discover_layers(model)
        handle=adapter.ModelHandle(model,CharacterTokenizer(),'test','cpu',path,layers)
        score=score_contrast(handle,'Color','red apple','green apple')
        self.assertEqual(score['positive_tokens'],10)
        self.assertEqual(score['negative_tokens'],12)
        self.assertNotEqual(score['positive_logp'],score['negative_logp'])

    def test_empty_failed_and_incomplete_evidence_never_confirms(self):
        good={k:{'status':'ok','passed':True} for k in ['necessity','completeness','minimality','counterexamples']}
        self.assertEqual(verification.evidence_gate([(0,0)],good)[0],'Confirmed')
        for status in ('incomplete','error'):
            records={**good,'completeness':{'status':status,'passed':True,'digest':'COMPLETE'}}
            self.assertNotEqual(verification.evidence_gate([(0,0)],records)[0],'Confirmed')
        self.assertNotEqual(verification.evidence_gate([],good)[0],'Confirmed')
        malicious_vote=HeuristicBackend(lambda *args:{'action':'Confirmed','reasoning':'fluent unsupported claim'})
        with patch('automechinterp.agents.judge.make_backend',return_value=malicious_vote):
            self.assertEqual(adjudicate([(0,0)],{'exclusion_digest':'INCOMPLETE'},'heuristic')['verdict'],'Speculative')

    def test_joint_complement_and_candidate_minimality_are_actual_coalitions(self):
        observed=[]
        def measure(handle,prompts,heads):
            group=frozenset(map(tuple,heads));observed.append(group)
            # Either backup alone suffices: only their simultaneous loss damages performance.
            value=0. if {(0,0),(0,1)}.issubset(group) else 1.
            return [value]*len(prompts)
        prompts=[(f'p{i}','a','b') for i in range(20)]
        with patch('automechinterp.tools.verification._measure',side_effect=measure):
            result=verification.completeness(None,[(1,0)],[(0,0),(0,1),(1,0)],prompts)
            self.assertFalse(result['passed'])
            self.assertIn(frozenset({(0,0),(0,1)}),observed)
            observed.clear()
            verification.minimality(None,[(0,0)],[(0,0),(0,1),(1,0)],prompts)
            self.assertIn(frozenset({(0,0),(0,1),(1,0)}),observed)

    def test_small_model_gold_is_not_used_for_other_checkpoints(self):
        result={'flagged_layers':[7,8,9,10,11],'claimed_heads':[(9,9)]}
        self.assertTrue(score_against_ground_truth(result,12,'gpt2')['ground_truth_valid'])
        self.assertFalse(score_against_ground_truth(result,48,'gpt2-xl')['ground_truth_valid'])

    def test_ratios_keep_negative_and_above_one_but_reject_zero_gap(self):
        self.assertEqual(normalized_effect([2,2],[1,1])['estimate'],2)
        self.assertEqual(normalized_effect([-1,-1],[1,1])['estimate'],-1)
        self.assertIsNone(normalized_effect([1,1],[0,0])['estimate'])

    def test_stop_decision_does_not_spend_tool_budget(self):
        budget=ToolCallBudget([5],5)
        agent=Agent(HeuristicBackend(lambda *args:{'action':'stop'}),{},budget)
        agent.run()
        self.assertEqual(budget.global_remaining,[5])

    def test_binomial_interval_does_not_claim_certainty_from_eight_successes(self):
        result=proportion(8,8)
        self.assertLess(result['ci95'][0],.70)
        self.assertGreater(proportion(0,8)['ci95'][1],.30)

    def test_hook_cleanup_on_exception(self):
        cfg=GPT2Config(vocab_size=64,n_embd=32,n_layer=2,n_head=4)
        model=AutoModelForCausalLM.from_config(cfg).eval()
        path,layers=adapter._discover_layers(model)
        handle=adapter.ModelHandle(model,CharacterTokenizer(),'test','cpu',path,layers)
        with self.assertRaises(RuntimeError):
            with ablate_heads(handle,[(0,0)]):
                raise RuntimeError('simulated failed model forward')
        self.assertFalse(adapter._find_attn_out_proj(layers[0])._forward_pre_hooks)

    def test_dataset_validation_rejects_cross_split_leakage(self):
        data=build_ioi_calibration_dataset()
        self.assertTrue(validate_dataset(data)['valid'])
        data['splits']['test'].append(data['splits']['discovery'][0])
        self.assertFalse(validate_dataset(data)['valid'])

    def test_model_loader_rejects_all_nonfinite_backends(self):
        model=AutoModelForCausalLM.from_config(GPT2Config(vocab_size=64,n_embd=32,n_layer=2,n_head=4))
        tokenizer=CharacterTokenizer();tokenizer.pad_token='pad'
        with patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer), \
             patch('transformers.AutoModelForCausalLM.from_pretrained',return_value=model), \
             patch('automechinterp.tools.adapter._logits_are_finite',return_value=False):
            with self.assertRaises(FloatingPointError):adapter.register_model('fixture')


if __name__=='__main__':
    unittest.main()
