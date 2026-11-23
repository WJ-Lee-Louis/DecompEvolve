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
for directory in ("src", "experiments/api_baseline", "experiments/gepa", "data/icml"):
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
def run(method: str, provider: str, run_id: str = "", model: str = "", split: str = "test",
        limit: int = 0, max_metric_calls: int = 0, seed: int = 42, reflection_provider: str = "",
        reflection_model: str = "", parallelism: int = 0, dry_run: bool = False, resume: bool = False):
    deadline = monotonic() + SESSION_TIMEOUT - CHECKPOINT_MARGIN
    from experiments.api_baseline.evaluate import PAUSED, main as baseline
    from experiments.gepa.optimize import main as gepa

    configs = {name: f"experiments/api_baseline/configs/optimind_{name}.json" for name in ("openai", "exaone")}
    if provider not in configs or (reflection_provider and reflection_provider not in configs):
        raise ValueError("Choose provider openai or exaone")
    if method not in ("baseline", "gepa"):
        raise ValueError("Choose baseline or gepa. DecompEvolve has no experiment entrypoint yet.")
    if not dry_run and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_id):
        raise ValueError("Use a unique run ID of 1–100 letters, numbers, underscores, or hyphens")
    args = ["--config", configs[provider]]
    if model:
        args.extend(["--model", model])
    if not dry_run:
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
    runner = baseline if method == "baseline" else gepa
    if dry_run:
        return runner(args)
    try:
        result = runner(args, checkpoint=volume.commit, deadline=deadline)
        if result == PAUSED:
            print(f"Checkpoint saved. Resume {method}/{run_id} with the same options and --resume.", flush=True)
        elif result:
            raise RuntimeError("Experiment interrupted. Inspect the saved run records before resuming.")
    finally:
        volume.commit()
