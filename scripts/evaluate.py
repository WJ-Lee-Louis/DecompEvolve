"""Evaluate a fixed prompt with one Responses API request per question."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from decompevolve.data import DATASET, ROOT, SPLITS, digest, load_dataset, load_splits
from decompevolve.evaluation import build_request, evaluate_one, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Config JSON path, relative to the project root")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit", type=int, help="Use the first N selected IDs for a smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Validate locally; no API calls or run files")
    parser.add_argument("--run-dir", help="Explicit output directory; default runs/baseline/<model>/<UTC timestamp>")
    parser.add_argument("--resume", action="store_true", help="Skip all recorded IDs in --run-dir; no automatic retries")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if config.get("reasoning_effort") not in ("low", "medium", "high", "xhigh", "max"):
        parser.error("A nonzero reasoning effort is required")
    if not isinstance(config.get("max_output_tokens"), int) or config["max_output_tokens"] <= 0:
        parser.error("max_output_tokens must be positive")
    if not isinstance(config.get("model"), str) or not config["model"].strip():
        parser.error("model must be a nonempty API model ID")
    if config.get("timeout_seconds", 0) <= 0:
        parser.error("timeout_seconds must be positive")
    prompt = (ROOT / config["prompt_file"]).read_text(encoding="utf-8")
    rows = load_dataset()
    splits = load_splits(rows)
    wanted = set(splits[args.split])
    selected = [row for row in rows if row["index"] in wanted]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        selected = selected[:args.limit]
    if not selected:
        parser.error("No selected records")
    spec = {
        "config": config, "split": args.split, "selected_ids": [r["index"] for r in selected],
        "dataset_sha256": digest(DATASET), "split_manifest": json.loads((SPLITS / "manifest.json").read_text()),
        "request_template": build_request("<QUESTION>", config, prompt),
        "evaluator_sha256": digest(ROOT / "src/decompevolve/evaluation.py"),
        "sdk_max_retries": 0,
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "model": config["model"], "split": args.split,
                          "questions": len(selected), "reasoning_effort": config["reasoning_effort"],
                          "max_output_tokens": config["max_output_tokens"], "tools": [], "tool_choice": "none",
                          "fingerprint": fingerprint, "api_calls": 0}, indent=2))
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Export OPENAI_API_KEY before a live evaluation. --dry-run needs no key.")
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    try:
        from openai import OpenAI
    except ImportError:
        parser.error("Install the project first: python -m pip install -e .")
    run_dir = ROOT / args.run_dir if args.run_dir else ROOT / "runs/baseline" / config["model"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    records = []
    if args.resume:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest["fingerprint"] != fingerprint:
            parser.error("Resume requires identical model, prompt, evaluator, dataset, and selection")
        path = run_dir / "predictions.jsonl"
        if path.exists():
            try:
                records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            except json.JSONDecodeError:
                parser.error("Run contains a partial JSONL record; inspect it before resuming")
        seen = [r["index"] for r in records]
        if len(seen) != len(set(seen)) or not set(seen) <= {r["index"] for r in selected}:
            parser.error("Invalid recorded IDs in resume directory")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "manifest.json").write_text(json.dumps({"fingerprint": fingerprint, **spec}, indent=2) + "\n", encoding="utf-8")
    client = OpenAI(max_retries=0, timeout=config["timeout_seconds"])
    seen = {r["index"] for r in records}
    print("Run directory:", run_dir, flush=True)
    failed = False
    try:
        with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            for row in selected:
                if row["index"] in seen:
                    continue
                record = evaluate_one(client, row, config, prompt)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(f"[{len(records)}/{len(selected)}] index={row['index']} {record['status']}", flush=True)
                if record["status"] == "api_error":
                    failed = True
                    print("Stopped after API error; check error_type/http_status in predictions.jsonl.", file=sys.stderr)
                    break
    except KeyboardInterrupt:
        failed = True
        print("Interrupted; recorded results are retained.", file=sys.stderr)
    finally:
        summary = summarize(records, len(selected))
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        client.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failed or any(r["status"] == "api_error" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
