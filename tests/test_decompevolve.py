import contextlib
import io
import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.api_baseline import evaluate, providers
from experiments.decompevolve import optimize
from experiments.decompevolve.curriculum import Curriculum, read_log
from experiments.gepa.optimize import append_log
from experiments.decompevolve.diagnosis import diagnose


class DecompEvolveTests(unittest.TestCase):
    def test_evidence_and_split_guards(self):
        record = dict(split="train", status="incorrect", question="Make whole units.", parsed_answer=4,
                      expected_answer=2, output_text="Answer: 4")
        self.assertEqual(diagnose(record, None)["stage"], "unknown")
        record["output_text"] = "I relaxed integrality.\nAnswer: 4"
        item = dict(stage="domain", evidence="I relaxed integrality.", condition="whole units", reason="Whole units required.",
                    check="Check variable domains.")
        self.assertEqual(diagnose(record, lambda _: json.dumps(item)), item)
        for raw in ("invalid", json.dumps(dict(item, evidence="Answer: 4")), "[]"):
            self.assertEqual(diagnose(record, lambda _: raw)["stage"], "unknown")
        for status, stage in [("timeout", "execution"), ("format_error", "format")]:
            self.assertEqual(diagnose(dict(record, status=status), None)["stage"], stage)
        with self.assertRaises(ValueError):
            diagnose(dict(record, split="val"), None)

    @unittest.skipUnless(find_spec("gepa"), "Install the gepa extra")
    def test_adaptive_levels_and_restart(self):
        from gepa.core.data_loader import ListDataLoader

        train = [dict(index=i+1, difficulty=d) for i, d in enumerate(("Easy", "Medium", "Hard"))]
        author = SimpleNamespace(generate=lambda diagnosis, question, level, key, seen:
                                 dict(question=f"test probe {key}", answer=1, fingerprint=key))
        for score, next_level in [(0, 1), (1, 3)]:
            with self.subTest(score=score), tempfile.TemporaryDirectory() as temp:
                path = Path(temp)
                loader = ListDataLoader(train)
                sampler = Curriculum(train, 42, path, loader, author)
                def batch(iteration):
                    return sampler.next_minibatch_ids(loader, SimpleNamespace(i=iteration))

                target = dict(index=1, diagnosis=dict(stage="constraints"))
                append_log(sampler.diagnostics, target)
                original = batch(0)
                self.assertEqual(set(original), {0, 1, 2})
                target["diagnosis"]["practice_skill"] = "proportional_link"
                append_log(sampler.diagnostics, target)
                first = batch(1)
                self.assertEqual(sampler.history[-1]["level"], 2)
                append_log(sampler.results, dict(indices=[-1], scores=[score]))
                loader = ListDataLoader(train)
                sampler = Curriculum(train, 42, path, loader, author)
                self.assertEqual(batch(0), original)
                self.assertEqual(batch(1), first)
                batch(2)
                self.assertEqual(sampler.history[-1]["level"], next_level)
                append_log(sampler.results, dict(indices=[-2], scores=[score]))
                self.assertEqual(set(batch(3)), {0, 1, 2})

    @unittest.skipUnless(find_spec("gepa"), "Install the gepa extra")
    def test_real_gepa_loop_with_mock_models(self):
        diagnosis = dict(stage="constraints", practice_skill="proportional_link", evidence="I omitted the linking constraint.",
                         reason="The response discarded a dependency.", check="Audit each linking constraint.")
        def reflect(client, config, prompt):
            self.assertEqual(config["model"], "gpt-5.4-nano")
            if prompt.startswith("Create one NEW"):
                self.assertIn(diagnosis["check"], prompt)
                return json.dumps(dict(names=["regular supply","certified supply"], upper=[12,12],
                    direction="min", integer=False, value=[2,5], target=1, rows=[
                    dict(kind="resource",unit="tons",a=[1,1],sense=">=",rhs=12),
                    dict(kind="ratio",a=[1,-2],sense="<=",rhs=0)])), {"total_tokens":100}
            if prompt.startswith("Review this LP/IP"):
                question = json.loads(prompt.splitlines()[-1])["question"]
                return json.dumps(dict(diagnosis, condition=question[:30])), {}
            self.assertIn(diagnosis["check"], prompt)
            return "```\nRevised instructions.\n```", {}

        def task(client, row, config, prompt):
            self.assertEqual(config["model"], "gpt-5.4-nano")
            correct = row["index"] < 0
            return dict(index=row["index"], question=row["question"], expected_answer=row["answer"],
                        status="correct" if correct else "incorrect", correct=correct,
                        parsed_answer=row["answer"] if correct else None,
                        output_text=diagnosis["evidence"] + "\nAnswer: 0", latency_seconds=0, usage={})

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            argv = ["--run-dir", str(run), "--max-metric-calls", "58"]
            with (patch.object(providers, "create_client", return_value=contextlib.nullcontext()),
                  patch.object(providers, "reflect", side_effect=reflect),
                  patch.object(providers, "evaluate_wave", side_effect=lambda c, rows, cfg, p:
                               [task(c, row, cfg, p) for row in rows]),
                  patch.object(evaluate, "create_client"), patch.object(evaluate, "evaluate_task", side_effect=task),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(optimize.main(argv), 0)
                self.assertEqual(optimize.main(argv + ["--resume"]), 0)
            probes = list(read_log(run / "synthetic.jsonl"))
            batches = list(read_log(run / "train_batches.jsonl"))
            self.assertTrue(probes)
            self.assertEqual(len(batches), 3)
            self.assertTrue(all(sum(i < 0 for i in b["indices"]) <= 1 for b in batches))
            logs = list(read_log(run / "rollouts.jsonl"))
            for i, batch in enumerate(batches):
                self.assertEqual([r["index"] for r in logs[23+6*i:26+6*i]], batch["indices"])
            sources = {r["index"] for r in logs if r["split"] == "train" and r["index"] > 0}
            self.assertTrue(all(p["source"] in sources for p in probes))
            self.assertFalse(any(r["index"] < 0 for r in logs if r["split"] != "train"))
