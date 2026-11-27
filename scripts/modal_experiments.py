"""Run DecompEvolve experiment entrypoints on a detached Modal CPU Function."""
import os
import re
from pathlib import Path
from time import monotonic

import modal

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = "/workspace"
VOLUME_PATH = "/experiment-results"
SESSION_TIMEOUT = 24 * 60 * 60
CHECKPOINT_MARGIN = 15 * 60
app = modal.App("decompevolve-experiments")
volume = modal.Volume.from_name("decompevolve-results", create_if_missing=True)
secret = modal.Secret.from_name("decompevolve-api-keys")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"), optional_dependencies=["gepa"])
    .env({"PYTHONPATH": f"{WORKSPACE}/src:{WORKSPACE}", "PYTHONUNBUFFERED": "1"})
)
ignored = ["**/__pycache__/**", "**/.DS_Store", "**/.env", "**/.env.*"]
for directory in ("src", "experiments/api_baseline", "experiments/gepa", "experiments/decompevolve", "data/icml"):
    image = image.add_local_dir(ROOT / directory, f"{WORKSPACE}/{directory}", ignore=ignored)
for file in ("datasets_IndustryOR_OptiMind_int.csv", "difficulty.csv"):
    relative = f"data/industryor_optimind/{file}"
    image = image.add_local_file(ROOT / relative, f"{WORKSPACE}/{relative}")
image = image.add_local_dir(
    ROOT / "data/industryor_optimind/splits", f"{WORKSPACE}/data/industryor_optimind/splits", ignore=ignored,
)


@app.function(image=image, secrets=[secret], timeout=60)
def smoke():
    from benchmark_support.data import load_experiment_dataset

    _, splits, _ = load_experiment_dataset("industryor_optimind")
    print({"remote": not modal.is_local(), "split_sizes": {name: len(ids) for name, ids in splits.items()},
           "openai_key_present": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
           "exaone_key_present": bool(os.environ.get("EXAONE_API_KEY", "").strip())})


@app.function(image=image, secrets=[secret], volumes={VOLUME_PATH: volume}, cpu=1, memory=512,
              retries=0, nonpreemptible=True, timeout=SESSION_TIMEOUT)
def run(method: str, provider: str = "openai", run_id: str = "", model: str = "", split: str = "test",
        limit: int = 0, max_metric_calls: int = 0, seed: int = 42, reflection_provider: str = "",
        reflection_model: str = "", parallelism: int = 0, dry_run: bool = False, resume: bool = False,
        diagnosis_only: bool = False, auto_continue: bool = False, author_token_budget: int = 0):
    deadline = monotonic() + SESSION_TIMEOUT - CHECKPOINT_MARGIN
    from experiments.api_baseline.evaluate import PAUSED, main as baseline
    from experiments.gepa.optimize import main as gepa

    from experiments.decompevolve.optimize import main as decompevolve

    configs = {name: f"experiments/api_baseline/configs/optimind_{name}.json" for name in ("openai", "exaone")}
    configs["openai"] = "experiments/api_baseline/configs/optimind_nano.json"
    if provider not in configs or (reflection_provider and reflection_provider not in configs):
        raise ValueError("Choose provider openai or exaone")
    if method not in ("baseline", "gepa", "decompevolve"):
        raise ValueError("Choose baseline, gepa, or decompevolve.")
    if not dry_run and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_id):
        raise ValueError("Use a unique run ID of 1–100 letters, numbers, underscores, or hyphens")
    args = ["--config", configs[provider]]
    if author_token_budget:
        if method != "decompevolve" or diagnosis_only:
            raise ValueError("nano author requires the full DecompEvolve method")
        args.extend(["--author-token-budget", str(author_token_budget)])
    if diagnosis_only:
        if method != "decompevolve":
            raise ValueError("--diagnosis-only requires --method decompevolve")
        args.append("--diagnosis-only")
    if model:
        args.extend(["--model", model])
    if not dry_run:
        volume.reload()
        target = Path(VOLUME_PATH) / method / run_id
        args.extend(["--run-dir", str(target)])
        print(f"Results Volume: decompevolve-results /{method}/{run_id}", flush=True)
    if dry_run:
        args.append("--dry-run")
    if resume:
        args.append("--resume")
    if parallelism:
        args.extend(["--parallelism", str(parallelism)])
    if method == "baseline":
        if max_metric_calls or reflection_provider or reflection_model:
            raise ValueError("Rollout budget and reflection model options apply to GEPA")
        args.extend(["--split", split])
        if limit:
            args.extend(["--limit", str(limit)])
    else:
        if split != "test" or limit:
            raise ValueError("GEPA uses the full fixed Train, Validation, and Test splits")
        if max_metric_calls:
            args.extend(["--max-metric-calls", str(max_metric_calls)])
        if reflection_provider:
            args.extend(["--reflection-config", configs[reflection_provider]])
        if reflection_model:
            args.extend(["--reflection-model", reflection_model])
        args.extend(["--seed", str(seed)])
    runner = {"baseline": baseline, "gepa": gepa, "decompevolve": decompevolve}[method]
    if dry_run:
        return runner(args)
    try:
        result = runner(args, checkpoint=volume.commit, deadline=deadline)
        if result == PAUSED:
            if auto_continue:
                volume.commit()
                followup = run.spawn(method=method, provider=provider, run_id=run_id, model=model, split=split,
                                     limit=limit, max_metric_calls=max_metric_calls, seed=seed,
                                     reflection_provider=reflection_provider, reflection_model=reflection_model,
                                     parallelism=parallelism, resume=True, diagnosis_only=diagnosis_only,
                                     auto_continue=True, author_token_budget=author_token_budget)
                print(f"Checkpoint saved. Continuing {method}/{run_id} in {followup.object_id}", flush=True)
            else:
                print(f"Checkpoint saved. Resume {method}/{run_id} with the same options and --resume.", flush=True)
        elif result:
            raise RuntimeError("Experiment interrupted. Inspect the saved run records before resuming.")
    finally:
        volume.commit()


@app.function(image=image, secrets=[secret], volumes={VOLUME_PATH: volume}, timeout=900)
def author_pilot(pilot_id: str):
    """Calibrate author tokens on two fixed Train-only requests before the main run."""
    import json
    from benchmark_support.data import load_experiment_dataset
    from experiments.api_baseline import providers
    from experiments.gepa.optimize import read_config
    from experiments.decompevolve.author import AUTHOR_EFFORT, AUTHOR_TIMEOUT, NanoAuthor, OUTPUT_LIMIT
    from experiments.decompevolve.curriculum import read_log
    from experiments.decompevolve.diagnosis import PRACTICE_SKILLS

    if not re.fullmatch(r"[A-Za-z0-9_-]+", pilot_id):
        raise ValueError("Invalid pilot ID")
    rows, splits, _ = load_experiment_dataset("industryor_optimind")
    train = {r["index"]: r["question"] for r in rows if r["index"] in splits["train"]}
    fixtures = [(13, "constraints", "at least twice", "I used only the total acreage constraint."),
                (88, "domain", "whole units", "I allowed fractional production.")]
    config = read_config("experiments/api_baseline/configs/optimind_nano.json")
    config.update(max_output_tokens=OUTPUT_LIMIT, timeout_seconds=AUTHOR_TIMEOUT, reasoning_effort=AUTHOR_EFFORT)
    path = Path(VOLUME_PATH) / "author-pilots" / pilot_id
    path.mkdir(parents=True, exist_ok=False)
    passed = []
    with providers.create_client(config) as client:
        author = NanoAuthor(path, lambda prompt: providers.reflect(client, config, prompt), 90000, volume.commit)
        for index, stage, phrase, simulated in fixtures:
            assert phrase in train[index]
            finding = dict(stage=stage, condition=phrase, evidence=simulated,
                           reason="The model omitted the quoted requirement.",
                           check="Translate and test the omitted requirement.", practice_skill=PRACTICE_SKILLS[stage])
            passed.append(bool(author.generate(finding, train[index], 2, str(index), set())))
    finished = [e for e in read_log(path/"authoring.jsonl") if e["status"]=="finished"]
    summary = dict(train_only=True, simulated_errors=True, accepted=passed, attempts=len(finished),
                   charged_tokens=sum(e["charged_tokens"] for e in finished))
    (path/"summary.json").write_text(json.dumps(summary))
    volume.commit()
    return summary
