"""Count actual requests and reported tokens by role from an optimizer run."""
import argparse
import json
from pathlib import Path


def summarize(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    roles = {role: dict(calls=0, missing_usage=0, input_tokens=0, output_tokens=0,
                        cached_input_tokens=0, reasoning_tokens=0)
             for role in ("optimization_task", "mutation", "diagnosis", "author", "test")}
    for filename, default_role in [("rollouts.jsonl", "optimization_task"),
                                   ("reflections.jsonl", "mutation"),
                                   ("authoring.jsonl", "author"),
                                   ("test/predictions.jsonl", "test")]:
        path = run_dir / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if default_role == "author" and record["status"] == "reserved":
                    continue
                row = roles[record.get("purpose", default_role)]
                row["calls"] += 1
                usage = record.get("usage") or {}
                if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
                    row["missing_usage"] += 1
                for field in ("input_tokens", "output_tokens"):
                    row[field] += usage.get(field) or 0
                row["cached_input_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
                row["reasoning_tokens"] += (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    return {"max_metric_calls": manifest["max_metric_calls"], "roles": roles}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(summarize(parser.parse_args().run_dir), indent=2))
