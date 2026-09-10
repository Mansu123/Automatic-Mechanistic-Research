import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from test_colab_bundle import fixture
from automechinterp.eval.matrix_rubric import IDS,health,validate_scores,validate_judgment
from automechinterp.eval.agreement import quadratic_kappa,compare
from automechinterp.eval.seap_evaluation import evaluate_seap,summarize_interactions
from automechinterp.eval.report_complete import write_report,coverage_records
from automechinterp.techniques.editing import _erasure_factors,leace
from automechinterp.behaviors import ioi_behavior
from llm_review import run_judge


class MatrixWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_eight_metrics_required_and_critical_failures_not_hidden_by_average(self):
        self.assertEqual(health(dict.fromkeys(IDS,8))["status"],"pass")
        scores=dict.fromkeys(IDS,10);scores["causal_validity"]=6
        self.assertEqual(health(scores)["status"],"needs_review")
        self.assertEqual(health(dict.fromkeys(IDS,10),False)["status"],"needs_review")
        for invalid in [dict.fromkeys(IDS,0),dict.fromkeys(IDS,11),dict.fromkeys(IDS,True),dict.fromkeys(IDS,7.5),{}]:
            with self.assertRaises(ValueError):validate_scores(invalid)

    def test_quadratic_kappa_matches_independent_library_and_constant_is_undefined(self):
        a=[1,2,3,4,5,6,7,8,9,10];b=[1,3,3,4,6,5,7,9,10,10]
        self.assertAlmostEqual(quadratic_kappa(a,b),cohen_kappa_score(a,b,weights="quadratic",labels=list(range(1,11))),places=12)
        self.assertIsNone(quadratic_kappa([8]*10,[8]*10))

    def test_judge_cannot_validate_on_small_or_constant_samples(self):
        human={str(i):dict.fromkeys(IDS,i%10+1) for i in range(20)}
        meta={i:{"model":"gpt2","angle":1} for i in human}
        result=compare(human,human,meta,bootstrap=30)
        self.assertEqual(result["status"],"unvalidated");self.assertAlmostEqual(result["macro_qwk"],1)
        human={f"{m}_{a}":dict.fromkeys(IDS,(m+a)%10+1) for m in range(22) for a in range(1,26)}
        meta={f"{m}_{a}":{"model":str(m),"angle":a} for m in range(22) for a in range(1,26)}
        self.assertEqual(compare(human,human,meta,bootstrap=30)["status"],"validated")

    def test_openrouter_exact_model_strict_evidence_and_no_score_fabrication(self):
        result={"metrics":{k:{"score":8,"reason":"Supported by supplied evidence","evidence_ids":["E000"]} for k in IDS}}
        response={"choices":[{"finish_reason":"stop","message":{"content":json.dumps(result)}}],"model":"z-ai/glm-5.2","usage":{"total_tokens":50}}
        item={"angle":25,"item_type":"report","element":"whole_report","execution_status":"completed"}
        with patch.object(run_judge,"request_json",return_value=response) as call:
            judged=run_judge.judge_item(item,{"E000":{"example":"test fixture"}})
            self.assertEqual(call.call_args.args[1]["model"],"z-ai/glm-5.2:free")
            self.assertEqual(judged["scores"],dict.fromkeys(IDS,8))
        result["metrics"]["faithfulness"]["evidence_ids"]=["invented"]
        with self.assertRaises(ValueError):validate_judgment(result,{"E000"})

    def test_full_layer_and_unexecuted_tools_are_visible_without_fabricated_effects(self):
        result={"n_layers":3,"execution_status":"completed","layer_states":{0:{"fraction_recovered":.1}},
                "agent_runs":[{"agent":"LayerAgent0","errors":[],"telemetry":[{"tool":"patch_layer","status":"ok","seconds":1,"observation":"full measurement"}],"available_tools":["patch_layer"],"evidence":["full measurement"]}],"verdict":None}
        task={"behavior":"fixture","category":"Angle 25: History"}
        text=write_report(task,result,"test_model")
        self.assertIn("L002",text);self.assertIn("full measurement",text)
        rows=coverage_records(result)
        self.assertTrue(any(r["agent"]=="LayerAgent2" and r["status"]=="not_run" for r in rows))

    def test_thin_svd_erasure_matches_dense_row_operator_and_bf16_hook(self):
        torch.manual_seed(3);X=torch.randn(12,20,dtype=torch.double);y=torch.tensor([1]*6+[0]*6)
        mu,left,right=_erasure_factors(X,y)
        xc=X-X.mean(0);sigma=xc.T@xc/len(X)+1e-3*torch.eye(20,dtype=torch.double)
        val,V=torch.linalg.eigh(sigma);W=(V*val.rsqrt())@V.T;Wi=(V*val.sqrt())@V.T
        delta=(xc[y==1].mean(0)-xc[y==0].mean(0))@W;u=delta/delta.norm()
        P=W@(torch.eye(20,dtype=torch.double)-u[:,None]*u[None,:])@Wi
        expected=xc@P+X.mean(0)
        actual=X.float()-((X.float()-mu)@left)[:,None]*right
        np.testing.assert_allclose(actual.numpy(),expected.numpy(),atol=2e-5)
        h=fixture(dtype=torch.bfloat16)
        result=leace(h,0,["good "+str(i) for i in range(3)],["bad "+str(i) for i in range(3)],"answer","yes","no")
        self.assertTrue(np.isfinite(result["margin_after"]))

    def test_seap_exact_formula_signed_and_zero_reference_is_not_perfect(self):
        h=fixture();t=ioi_behavior(h)
        result=evaluate_seap(h,t,candidate_heads=3,evaluation_pairs=2)
        self.assertEqual(result["status"],"completed",result)
        self.assertEqual(len(result["rows"]),6)
        for row in result["rows"]:
            self.assertAlmostEqual(row["exact"],row["f_ij"]-row["f_i"]-row["f_j"]+row["f_empty"])
        zeros=summarize_interactions([{"prompt_index":0,"exact":0,"approximation":0}])
        self.assertIsNone(zeros["score_0to100"])
        perfect=summarize_interactions([{"prompt_index":0,"exact":-2,"approximation":-2},{"prompt_index":1,"exact":3,"approximation":3}])
        self.assertEqual(perfect["score_0to100"],100)


if __name__=="__main__":unittest.main()
