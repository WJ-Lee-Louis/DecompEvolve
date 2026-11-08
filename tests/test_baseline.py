import json
import tempfile
import unittest
import contextlib
import io
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from decompevolve.data import ROOT, SPLITS, allocate_groups, load_dataset, load_splits
from decompevolve.evaluation import build_request, evaluate_one, parse_answer, summarize


class FakeResponse:
    def __init__(self, text, status="completed", output=None):
        self.output_text = text
        self.raw = {"id": "fake", "status": status, "output": output or [],
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                              "output_tokens_details": {"reasoning_tokens": 15}}}

    def model_dump(self, **kwargs):
        return self.raw


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response
        self.error = error
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response

    def close(self):
        pass


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/baseline_luna.json").read_text())
        self.row = {"index": 1, "question": "An optimization problem", "answer": -200}

    def test_last_line_and_signed_integer(self):
        for text, expected in [("Answer: -200", -200), ("Summary\nAnswer: +0\n\n", 0),
                               ("Answer: 1\nUpdated\nAnswer: 2", 2)]:
            self.assertEqual(parse_answer(text), expected)
        for text in ["", "Answer: 3900.0", "Answer: 3,900", "Answer: 12 units", "Answer: 1\nextra", "Answer: NaN", "Answer: 2.5"]:
            self.assertIsNone(parse_answer(text))

    def test_single_request_no_tools_no_gold(self):
        client = FakeClient(FakeResponse("Answer: -200"))
        record = evaluate_one(client, self.row, self.config, "Solve.")
        self.assertEqual(record["status"], "correct")
        self.assertEqual(len(client.calls), 1)
        request = client.calls[0]
        self.assertEqual(request["tools"], [])
        self.assertEqual(request["tool_choice"], "none")
        self.assertFalse(request["store"])
        self.assertEqual(request["reasoning"], {"effort": "medium"})
        self.assertNotIn("-200", json.dumps(request))

    def test_incomplete_does_not_receive_credit(self):
        record = evaluate_one(FakeClient(FakeResponse("Answer: -200", status="incomplete")), self.row, self.config, "Solve.")
        self.assertEqual(record["status"], "incomplete")
        self.assertFalse(record["correct"])

    def test_tool_request_is_never_executed_or_credited(self):
        response = FakeResponse("Answer: -200", output=[{"type": "function_call", "name": "python"}])
        record = evaluate_one(FakeClient(response), self.row, self.config, "Solve.")
        self.assertEqual(record["status"], "unexpected_tool_call")
        self.assertFalse(record["correct"])

    def test_incorrect_format_and_refusal(self):
        for response, expected in [(FakeResponse("Answer: 4"), "incorrect"),
                                   (FakeResponse("-200"), "format_error"),
                                   (FakeResponse("", output=[{"type": "message", "content": [{"type": "refusal"}]}]), "refusal")]:
            record = evaluate_one(FakeClient(response), self.row, self.config, "Solve.")
            self.assertEqual(record["status"], expected)

    def test_api_error_once_and_no_secret_dump(self):
        client = FakeClient(error=RuntimeError("sensitive-example"))
        record = evaluate_one(client, self.row, self.config, "Solve.")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(record["status"], "api_error")
        self.assertNotIn("sensitive-example", json.dumps(record))

    def test_summary_denominator_and_token_accounting(self):
        record = evaluate_one(FakeClient(FakeResponse("Answer: -200")), self.row, self.config, "Solve.")
        partial = summarize([record], 2)
        self.assertIsNone(partial["accuracy"])
        self.assertEqual(partial["remaining"], 1)
        complete = summarize([record], 1)
        self.assertEqual(complete["accuracy"], 1.0)
        self.assertEqual(complete["usage"]["output_tokens"], 20)
        self.assertEqual(complete["usage"]["reasoning_tokens"], 15)

    def test_real_splits_coverage_groups_and_known_variants(self):
        rows = load_dataset()
        splits = load_splits(rows)
        self.assertEqual({name: len(ids) for name, ids in splits.items()}, {"train": 222, "val": 50, "test": 100})
        member = {index: name for name, ids in splits.items() for index in ids}
        for family in [[148, 149, 358, 359], [178, 387], [181, 182, 392, 393], [153, 154, 363, 364], [357, 410]]:
            self.assertEqual(len({member[i] for i in family}), 1)
        self.assertEqual(len(rows), 372)

    def test_allocation_reproducible_and_indivisible(self):
        groups = [[1, 2], [3, 4], [5], [6], [7], [8]]
        first = allocate_groups(groups, 2, 3, 42)
        self.assertEqual(first, allocate_groups(groups, 2, 3, 42))
        self.assertEqual(len(first["val"]), 2)
        self.assertEqual(len(first["test"]), 3)
        member = {i: name for name, ids in first.items() for i in ids}
        for group in groups:
            self.assertEqual(len({member[i] for i in group}), 1)
        with self.assertRaises(ValueError):
            allocate_groups([[1, 2], [3, 4]], 1, 1, 42)

    def test_modified_or_overlapping_splits_rejected(self):
        rows = load_dataset()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for path in SPLITS.glob("*.json"):
                (directory / path.name).write_bytes(path.read_bytes())
            train = json.loads((directory / "train_ids.json").read_text())
            val = json.loads((directory / "val_ids.json").read_text())
            train[0] = val[0]
            (directory / "train_ids.json").write_text(json.dumps(train))
            with self.assertRaises(ValueError):
                load_splits(rows, directory=directory)

    def test_real_sdk_serialization_without_network(self):
        import httpx
        from openai import OpenAI
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "resp_mock", "object": "response", "created_at": 0,
                "status": "completed", "model": self.config["model"],
                "output": [{"id": "msg_mock", "type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "Answer: -200", "annotations": []}]}],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 15}},
            })

        with OpenAI(api_key="offline-test-only", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            result = evaluate_one(client, self.row, self.config, "Solve.")
        self.assertEqual(result["status"], "correct")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["tool_choice"], "none")

    def test_cli_resume_skips_recorded_requests(self):
        main = runpy.run_path(str(ROOT / "scripts/evaluate.py"))["main"]
        rows = load_dataset()
        test_ids = set(load_splits(rows)["test"])
        expected = next(row["answer"] for row in rows if row["index"] in test_ids)
        fake = FakeClient(FakeResponse(f"Answer: {expected}"))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            argv = ["evaluate.py", "--config", "configs/baseline_luna.json", "--limit", "1", "--run-dir", str(run_dir)]
            with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-only"}), patch("openai.OpenAI", return_value=fake), contextlib.redirect_stdout(io.StringIO()):
                with patch("sys.argv", argv):
                    self.assertEqual(main(), 0)
                with patch("sys.argv", argv + ["--resume"]):
                    self.assertEqual(main(), 0)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(len((run_dir / "predictions.jsonl").read_text().splitlines()), 1)
            self.assertEqual(json.loads((run_dir / "summary.json").read_text())["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
