import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from automechinterp import behaviors
from automechinterp.eval.agreement import compare
from automechinterp.eval.matrix_rubric import IDS, METRIC_BANDS, rubric_text
from automechinterp.eval.task_audit import audit_task
from automechinterp.eval.review_export import export_run, item_evidence
from test_colab_bundle import fixture
import run_colab


class ReleaseContractTests(unittest.TestCase):
    def test_full_catalog_is_angle_ordered_and_reordered_config_rejected(self):
        flattened=[b for a in range(1,26) for b in behaviors.ANGLE_BEHAVIORS[a]]
        self.assertEqual(behaviors.ALL_BEHAVIORS,flattened)
        config=run_colab.load_config(run_colab.REPO/"colab_config.json")
        config["behavior_ids"].reverse()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"config.json";p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):run_colab.load_config(p)

    def test_all_behaviors_have_split_integrity_and_metric_specific_controls(self):
        h=fixture()
        for builder in behaviors.ALL_BEHAVIORS:
            with self.subTest(behavior=builder.__name__):
                task=builder(h,seed=0)
                self.assertEqual(audit_task(h,task)["status"],"passed")
                self.assertEqual(task["dataset_hash"],builder(h,seed=0)["dataset_hash"])
        self.assertEqual(set(METRIC_BANDS),set(IDS))
        for angle in range(1,26):
            text=rubric_text(angle)
            for key in IDS:
                for band in METRIC_BANDS[key]:self.assertIn(band,text)

    def test_token_collision_context_overflow_and_leakage_rejected(self):
        h=fixture();task=behaviors.ioi_behavior(h)
        leaked={**task,"eval_pairs":task["discovery_pairs"]}
        with self.assertRaises(ValueError):audit_task(h,leaked)
        with patch.object(h.model.config,"n_positions",2):
            with self.assertRaises(ValueError):audit_task(h,task)

    def test_550_arbitrary_cells_are_not_a_22_by_25_validation(self):
        scores={str(i):dict.fromkeys(IDS,i%10+1) for i in range(550)}
        meta={str(i):{"model":str(i),"angle":1} for i in range(550)}
        self.assertEqual(compare(scores,scores,meta,bootstrap=30)["status"],"unvalidated")
        invalid={**scores,"0":dict.fromkeys(IDS,7.1)}
        with self.assertRaises(ValueError):compare(invalid,scores,meta,bootstrap=30)

    def test_export_includes_unexecuted_agents_tools_and_shared_evidence(self):
        h=fixture();task=behaviors.ioi_behavior(h)
        task={k:v for k,v in task.items() if k!="task_metric_fn"}
        config=run_colab.load_config(run_colab.REPO/"colab_config.json")
        ident=run_colab.identity("fixture",config["models"][0],"ioi_behavior",0)
        raw={"n_layers":2,"layer_states":{},"agent_runs":[],"execution_status":"completed"}
        payload={"identity":ident,"status":"completed","entry":{"task_definition":task,"model":"gpt2","raw_hierarchy_result":raw}}
        with tempfile.TemporaryDirectory() as temp, patch("automechinterp.eval.review_export._plot_matrix"):
            root=Path(temp)
            run_colab.atomic_json(root/"run_manifest.json",{"run_id":"fixture","config":config,"expected_jobs":4466})
            run_colab.save_checkpoint(root,None,payload)
            export_run(root)
            items=[json.loads(line) for line in (root/"review/review_items.jsonl").read_text().splitlines()]
            selected=next(i for i in items if i["element"]=="ComponentAgent1/run_eap")
            self.assertEqual(selected["execution_status"],"not_run")
            self.assertIn("E006",item_evidence(root,selected))
            self.assertTrue(any(i["element"]=="Orchestrator" for i in items))
            self.assertFalse(json.loads((root/"review/judge_validation_sample.json").read_text())["frozen"])


if __name__=="__main__":unittest.main()
