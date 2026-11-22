import asyncio
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from benchmark_support.data import load_experiment_dataset
from experiments.api_baseline import providers
from experiments.gepa import optimize

OPENAI_CONFIG = "experiments/api_baseline/configs/optimind_openai.json"
EXAONE_CONFIG = "experiments/api_baseline/configs/optimind_exaone.json"


class OptiMindTests(unittest.TestCase):
    def test_exaone_wire_format_and_grading(self):
        config = optimize.read_config(EXAONE_CONFIG)
        row = {"index": 1, "question": "Find the best feasible objective.", "answer": -12345}

        async def respond(request):
            self.assertEqual(str(request.url), providers.EXAONE_URL + "/chat/completions")
            payload = json.loads(request.content)
            self.assertEqual(payload["messages"][-1], {"role": "user", "content": row["question"]})
            self.assertNotIn(str(row["answer"]), json.dumps(payload))
            self.assertNotIn("tools", payload)
            self.assertEqual(payload["reasoning_effort"], "high")
            self.assertEqual(payload["max_tokens"], 65536)
            return httpx.Response(200, json={
                "id": "mock", "object": "chat.completion", "created": 0, "model": config["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Checked.\nAnswer: -12345"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50,
                          "reasoning_tokens": 25},
            })

        sdk = AsyncOpenAI(api_key="mock", base_url=providers.EXAONE_URL, max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        with providers.DeadlineClient(sdk, timeout=config["timeout_seconds"]) as client:
            record = providers.evaluate_task(client, row, config, "Solve this.")
        self.assertEqual(record["status"], "correct")
        self.assertEqual(record["usage"]["output_tokens_details"]["reasoning_tokens"], 25)

    @unittest.skipUnless(importlib.util.find_spec("gepa"), "Install the gepa extra")
    def test_gepa_mutation_skip_holdout_and_resume(self):
        rows, splits, _ = load_experiment_dataset("industryor_optimind")
        test_ids = set(splits["test"])
        test_questions = [row["question"] for row in rows if row["index"] in test_ids]
        for perfect in (False, True):
            with self.subTest(perfect=perfect), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "run"
                argv = ["--config", OPENAI_CONFIG, "--run-dir", str(target), "--max-metric-calls", "46"]

                def task(client, row, config, prompt):
                    correct = perfect or prompt.strip() == "Revised instructions."
                    return {
                        "index": row["index"], "question": row["question"], "expected_answer": row["answer"],
                        "correct": correct, "status": "correct" if correct else "incorrect",
                        "parsed_answer": row["answer"] if correct else 0, "output_text": "Mock visible trace.",
                        "usage": None, "latency_seconds": 0,
                    }

                def reflect(client, config, prompt):
                    self.assertIn("Mock visible trace.", prompt)
                    self.assertIn("Reference answer", prompt)
                    self.assertFalse(any(question in prompt for question in test_questions))
                    return "```\nRevised instructions.\n```", None

                with (
                    patch.object(providers, "create_client", return_value=contextlib.nullcontext()),
                    patch.object(optimize.evaluate, "create_client") as baseline_client,
                    patch.object(providers, "evaluate_wave", side_effect=lambda client, rows, config, prompt: [
                        task(client, row, config, prompt) for row in rows
                    ]),
                    patch.object(optimize.evaluate, "evaluate_task", side_effect=task) as test_task,
                    patch.object(providers, "reflect", side_effect=reflect) as reflector,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(optimize.main(argv, checkpoint=lambda: None), 0)
                    self.assertEqual(optimize.main(argv + ["--resume"]), 0)
                    self.assertEqual(reflector.call_count, 0 if perfect else 1)
                    self.assertEqual([call.args[1]["index"] for call in test_task.call_args_list], splits["test"])
                    self.assertEqual(baseline_client.return_value.close.call_count, 2)

                logs = [json.loads(line) for line in (target / "rollouts.jsonl").read_text().splitlines()]
                self.assertEqual(len(logs), 23 if perfect else 46)
                self.assertEqual({entry["difficulty"] for entry in logs[20:23]}, {"Easy", "Medium", "Hard"})
                self.assertEqual({entry["split"] for entry in logs}, {"train", "val"})
                if not perfect:
                    self.assertEqual([row["index"] for row in logs[20:23]], [row["index"] for row in logs[23:26]])
                    self.assertEqual((target / "best_prompt.txt").read_text().strip(), "Revised instructions.")
                summary = json.loads((target / "test" / "summary.json").read_text())
                self.assertEqual(summary["accuracy"], 1)
                self.assertEqual({key: val["correct"] for key, val in summary["by_difficulty"].items()},
                                 {"Easy": 11, "Medium": 15, "Hard": 4})

    def test_concurrent_requests_enforce_individual_deadlines(self):
        config = optimize.read_config(EXAONE_CONFIG)
        active = peak = 0

        async def respond(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            index = int(json.loads(request.content)["messages"][-1]["content"])
            try:
                await asyncio.sleep(1 if index == 1 else 0.005)
            finally:
                active -= 1
            return httpx.Response(200, json={
                "id": "mock", "object": "chat.completion", "created": 0, "model": config["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Answer: 7"}}],
            })

        sdk = AsyncOpenAI(api_key="mock", base_url=providers.EXAONE_URL, max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        rows = [{"index": index, "question": str(index), "answer": 7} for index in (1, 2, 3)]
        with providers.DeadlineClient(sdk, timeout=0.15) as client:
            records = providers.evaluate_wave(client, rows, config, "Solve.")
        self.assertEqual(peak, 3)
        self.assertEqual(active, 0)
        self.assertEqual([record["status"] for record in records], ["timeout", "correct", "correct"])
        with tempfile.TemporaryDirectory() as temporary:
            adapter = optimize.OptiMindAdapter(None, config, Path(temporary), 600, 4, None)
            batch = SimpleNamespace(trajectories=[dict(records[0], split="train")], scores=[0])
            feedback = adapter.make_reflective_dataset({}, batch, [optimize.COMPONENT])
        text = feedback[optimize.COMPONENT][0]["Feedback"]
        self.assertIn("Status: incorrect.", text)
        self.assertNotIn("timeout", text)

    def test_baseline_timeout_is_scored_and_paused_test_resumes_at_next_item(self):
        def task(client, row, config, prompt):
            status = "timeout" if task_mock.call_count == 1 else "correct"
            return {
                "index": row["index"], "status": status, "correct": status == "correct",
                "latency_seconds": 0, "usage": None,
            }

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "test"
            argv = ["--config", OPENAI_CONFIG, "--run-dir", str(target), "--limit", "2"]
            with (
                patch.object(optimize.evaluate, "create_client"),
                patch.object(optimize.evaluate, "evaluate_task", side_effect=task) as task_mock,
                patch.object(optimize.evaluate, "monotonic", side_effect=[0, 900]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(optimize.evaluate.main(argv, deadline=900), optimize.evaluate.PAUSED)
                self.assertEqual(task_mock.call_count, 1)
                self.assertEqual(optimize.evaluate.main(argv + ["--resume"]), 0)
            summary = json.loads((target / "summary.json").read_text())
            self.assertEqual(summary["status_counts"], {"timeout": 1, "correct": 1})
            self.assertEqual(summary["accuracy"], 0.5)

    @unittest.skipUnless(importlib.util.find_spec("gepa"), "Install the gepa extra")
    def test_gepa_pauses_before_iteration_and_resumes_without_repeating_seed_validation(self):
        def task(client, row, config, prompt):
            return {
                "index": row["index"], "question": row["question"], "expected_answer": row["answer"],
                "correct": True, "status": "correct", "parsed_answer": row["answer"], "output_text": "Trace.",
                "usage": None, "latency_seconds": 0,
            }

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "run"
            argv = ["--config", OPENAI_CONFIG, "--run-dir", str(target), "--max-metric-calls", "46"]
            with (
                patch.object(optimize, "monotonic", return_value=0),
                patch.object(providers, "create_client", return_value=contextlib.nullcontext()),
                patch.object(optimize.evaluate, "create_client"),
                patch.object(providers, "evaluate_wave", side_effect=lambda client, rows, config, prompt: [
                    task(client, row, config, prompt) for row in rows
                ]) as waves,
                patch.object(optimize.evaluate, "evaluate_task", side_effect=task),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(optimize.main(argv, deadline=4550), optimize.evaluate.PAUSED)
                self.assertTrue((target / "optimizer" / "gepa_state.bin").is_file())
                self.assertFalse((target / "best_prompt.txt").exists())
                self.assertEqual(waves.call_count, 5)
                self.assertEqual(optimize.main(argv + ["--resume"]), 0)
                self.assertEqual(waves.call_count, 6)
            logs = [json.loads(line) for line in (target / "rollouts.jsonl").read_text().splitlines()]
            self.assertEqual(len(logs), 23)
            self.assertEqual(sum(entry.get("stage") == "seed_validation" for entry in logs), 20)
