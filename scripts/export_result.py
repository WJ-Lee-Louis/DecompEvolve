"""Export a finished Baseline or GEPA run into the Git-tracked results directory."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def without_provider_response(value):
    """Keep experiment data while dropping embedded provider response objects."""
    if isinstance(value, dict):
        return {k: without_provider_response(v) for k, v in value.items() if k != "response"}
    if isinstance(value, list):
        return [without_provider_response(item) for item in value]
    return value


def export_result(source, results_root):
    source = Path(source).resolve()
    manifest = read_json(source / "manifest.json")
    is_gepa = "task" in manifest and "split_ids" in manifest
    method = "gepa" if is_gepa else "baseline"
    if "decompevolve" in manifest:
        method = "decompevolve"
    model = manifest["task" if is_gepa else "config"]["model"]
    for segment in (model, source.name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", segment):
            raise ValueError(f"Invalid result path segment: {segment!r}")

    relative = Path("test") if is_gepa else Path()
    evaluation = source / relative
    evaluation_manifest = read_json(evaluation / "manifest.json")
    summary = read_json(evaluation / "summary.json")
    predictions = read_jsonl(evaluation / "predictions.jsonl")
    indices = [record["index"] for record in predictions]
    status_counts = Counter(record["status"] for record in predictions)
    if (
        not predictions or summary["completed"] is not True or summary["remaining"] != 0
        or status_counts["api_error"]
        or summary["records_written"] != len(predictions) or summary["total_selected"] != len(predictions)
        or indices != evaluation_manifest["selected_ids"] or len(indices) != len(set(indices))
        or summary["correct"] != sum(record["correct"] for record in predictions)
        or summary["status_counts"] != status_counts
        or (is_gepa and (evaluation_manifest["split"] != "test" or set(indices) != set(manifest["split_ids"]["test"])))
    ):
        raise ValueError("Run is incomplete or its recorded results disagree")

    def json_bytes(data):
        return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    def jsonl_bytes(rows):
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")

    artifacts = {
        relative / "manifest.json": (evaluation / "manifest.json").read_bytes(),
        relative / "summary.json": (evaluation / "summary.json").read_bytes(),
        relative / "predictions.jsonl": jsonl_bytes(without_provider_response(predictions)),
    }
    if is_gepa:
        artifacts.update({
            Path("manifest.json"): (source / "manifest.json").read_bytes(),
            Path("best_prompt.txt"): (source / "best_prompt.txt").read_bytes(),
            Path("optimization.json"): json_bytes(without_provider_response(read_json(source / "optimization.json"))),
            Path("rollouts.jsonl"): jsonl_bytes(without_provider_response(read_jsonl(source / "rollouts.jsonl"))),
        })
        reflections = source / "reflections.jsonl"
        if reflections.is_file():
            artifacts[Path("reflections.jsonl")] = jsonl_bytes(without_provider_response(read_jsonl(reflections)))

    if method == "decompevolve":
        for name in ("diagnoses.jsonl", "authoring.jsonl", "synthetic.jsonl", "train_batches.jsonl", "original_batches.jsonl"):
            if (source / name).is_file():
                artifacts[Path(name)] = (source / name).read_bytes()

    target = Path(results_root) / method / model / source.name
    if target.exists():
        files = {p.relative_to(target) for p in target.rglob("*") if p.is_file()}
        if files == artifacts.keys() and all((target / name).read_bytes() == data for name, data in artifacts.items()):
            return target
        raise FileExistsError(f"A different export already exists at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as temporary:
        staged = Path(temporary) / "export"
        for name, data in artifacts.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        staged.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Local run directory, including one downloaded from Modal")
    args = parser.parse_args()
    try:
        target = export_result(args.run_dir, ROOT / "results")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
