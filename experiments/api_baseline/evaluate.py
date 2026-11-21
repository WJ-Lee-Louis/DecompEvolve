"""Evaluate a fixed prompt with one tool-free model request per question."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from benchmark_support.data import ROOT, digest, load_experiment_dataset
from benchmark_support.environment import load_environment

from .evaluation import summarize
from .providers import create_client, evaluate_task, task_request, validate_config

PAUSED = 75


def main(argv=None, checkpoint=None, deadline=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Config JSON path, relative to the project root")
    parser.add_argument("--model", help="Override the model ID within the selected provider")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit", type=int, help="Use the first N selected IDs for a smoke test")
    parser.add_argument("--dry-run", action="store_true", help="Validate locally; no API calls or run files")
    parser.add_argument("--run-dir", help="Explicit output directory; default runs/baseline/<model>/<UTC timestamp>")
    parser.add_argument("--prompt-file", help="Evaluate a saved prompt using the selected model and output contract")
    parser.add_argument("--resume", action="store_true", help="Skip all recorded IDs in --run-dir; no automatic retries")
    args = parser.parse_args(argv)
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.model:
        config["model"] = args.model
    if args.prompt_file:
        config["prompt_file"] = args.prompt_file
    dataset_name = config.get("dataset", "icml")
    try:
        provider = validate_config(config)
        rows, splits, split_manifest = load_experiment_dataset(dataset_name)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    prompt = (ROOT / config["prompt_file"]).read_text(encoding="utf-8")
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
        "dataset_sha256": split_manifest["dataset_sha256"], "split_manifest": split_manifest,
        "request_template": task_request("<QUESTION>", config, prompt),
        "evaluator_sha256": digest(Path(__file__).with_name("evaluation.py")),
        "sdk_max_retries": 0,
    }
    if "provider" in config:
        spec["provider_sha256"] = digest(Path(__file__).with_name("providers.py"))
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "model": config["model"], "provider": provider,
                          "dataset": dataset_name, "split": args.split,
                          "questions": len(selected), "reasoning_effort": config["reasoning_effort"],
                          "max_output_tokens": config["max_output_tokens"], "timeout_seconds": config["timeout_seconds"],
                          "tools": [], "tool_choice": "none",
                          "fingerprint": fingerprint, "api_calls": 0}, indent=2))
        return 0
    load_environment()
    if args.resume and not args.run_dir:
        parser.error("--resume requires --run-dir")
    run_dir = ROOT / args.run_dir if args.run_dir else ROOT / "runs/baseline" / config["model"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    records = []
    if args.resume:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        resume_spec = spec.copy()
        # Reconstruct historical path values while hashing the current evaluation inputs.
        previous_prompt = manifest.get("config", {}).get("prompt_file")
        if config["prompt_file"] == "experiments/api_baseline/prompts/baseline.txt" and previous_prompt in (
            "prompts/baseline.txt", "experiments/baseline/prompts/baseline.txt",
        ):
            resume_spec["config"] = {**config, "prompt_file": previous_prompt}
        previous_dataset = manifest.get("split_manifest", {}).get("dataset")
        if previous_dataset == "data/datasets_ICML_int.csv":
            resume_spec["split_manifest"] = {**spec["split_manifest"], "dataset": previous_dataset}
        if manifest["fingerprint"] != hashlib.sha256(json.dumps(resume_spec, sort_keys=True).encode()).hexdigest():
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
    try:
        client = create_client(config)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "manifest.json").write_text(json.dumps({"fingerprint": fingerprint, **spec}, indent=2) + "\n", encoding="utf-8")
    seen = {r["index"] for r in records}
    print("Run directory:", run_dir, flush=True)
    failed = False
    paused = False
    try:
        with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            for row in selected:
                if row["index"] in seen:
                    continue
                if deadline is not None and monotonic() + config["timeout_seconds"] > deadline:
                    paused = True
                    break
                record = evaluate_task(client, row, config, prompt)
                if "difficulty" in row:
                    record["difficulty"] = row["difficulty"]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
                print(f"[{len(records)}/{len(selected)}] index={row['index']} {record['status']}", flush=True)
                if checkpoint:
                    checkpoint()
                if record["status"] == "api_error":
                    failed = True
                    print("Stopped after API error; check error_type/http_status in predictions.jsonl.", file=sys.stderr)
                    break
    except KeyboardInterrupt:
        failed = True
        print("Interrupted; recorded results are retained.", file=sys.stderr)
    finally:
        summary = summarize(records, len(selected))
        if paused:
            summary["paused"] = True
        if dataset_name == "industryor_optimind":
            summary["note"] += " Timeouts and token-limit responses also score zero."
            summary["by_difficulty"] = {}
            for level in ("Easy", "Medium", "Hard"):
                total = sum(row["difficulty"] == level for row in selected)
                if total:
                    level_records = [record for record in records if record.get("difficulty") == level]
                    summary["by_difficulty"][level] = summarize(level_records, total)
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        client.close()
        if checkpoint:
            checkpoint()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if paused:
        return PAUSED
    return 1 if failed or any(r["status"] == "api_error" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
