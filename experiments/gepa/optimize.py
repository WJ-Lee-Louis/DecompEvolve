"""Run GEPA on the fixed OptiMind splits with a provider-backed, tool-free rollout."""
import argparse
import json
import random
import sys
from contextlib import ExitStack
from math import ceil
from pathlib import Path
from time import monotonic

from benchmark_support.data import ROOT, digest, load_experiment_dataset
from benchmark_support.environment import load_environment
from experiments.api_baseline import evaluate
from experiments.api_baseline import evaluation, providers

DIFFICULTIES = ("Easy", "Medium", "Hard")
COMPONENT = "instructions"
DEFAULT_METRIC_CALLS = 600
DEFAULT_PARALLELISM = 4
REFLECTION_TIMEOUT = 60
REFLECTION_MAX_TOKENS = 8192


class DifficultySampler:
    """Take one per difficulty, cycling through its shuffled train items."""

    def __init__(self, train, seed):
        self.pools = {level: [i for i, row in enumerate(train) if row["difficulty"] == level]
                      for level in DIFFICULTIES}
        if any(not pool for pool in self.pools.values()):
            raise ValueError("GEPA requires train items at all three difficulty levels")
        self.seed = seed

    def next_minibatch_ids(self, loader, state):
        sampled = []
        for level, pool in self.pools.items():
            cycle, offset = divmod(state.i, len(pool))
            deck = pool.copy()
            random.Random(f"{self.seed}:{level}:{cycle}").shuffle(deck)
            sampled.append(deck[offset])
        return sampled


def append_log(path, entry):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


class OptiMindAdapter:
    propose_new_texts = None

    def __init__(self, client, config, run_dir, max_calls, parallelism, checkpoint):
        self.client = client
        self.config = config
        self.log = run_dir / "rollouts.jsonl"
        self.max_calls = max_calls
        self.parallelism = parallelism
        self.checkpoint = checkpoint
        self.calls = 0
        self.initializing = True
        self.seed_records = {}
        if self.log.exists():
            with self.log.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    self.calls += 1
                    if record.get("stage") == "seed_validation" and record["status"] != "api_error":
                        self.seed_records[record["index"]] = record

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        initializing = self.initializing
        self.initializing = False
        results = self.seed_records.copy() if initializing else {}
        pending = [row for row in batch if row["index"] not in results]
        if self.calls + len(pending) > self.max_calls:
            raise RuntimeError("Remaining rollout budget is insufficient for this batch")
        for start in range(0, len(pending), self.parallelism):
            wave = pending[start:start + self.parallelism]
            records = providers.evaluate_wave(self.client, wave, self.config, candidate[COMPONENT])
            for row, record in zip(wave, records):
                record.update(split=row["split"], difficulty=row["difficulty"])
                if initializing:
                    record["stage"] = "seed_validation"
                append_log(self.log, record)
                self.calls += 1
                results[row["index"]] = record
            if self.checkpoint:
                self.checkpoint()
            failed = next((record for record in records if record["status"] == "api_error"), None)
            if failed:
                raise RuntimeError(f"Task API failed on item {failed['index']} ({failed['error_type']}, HTTP {failed['http_status']})")
        outputs = [results[row["index"]] for row in batch]
        return EvaluationBatch(
            outputs=outputs,
            scores=[float(record["correct"]) for record in outputs],
            trajectories=outputs if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        feedback = []
        for record, score in zip(eval_batch.trajectories, eval_batch.scores):
            if record["split"] != "train":
                raise ValueError("Only train items may enter reflection")
            status = "incorrect" if record["status"] in ("timeout", "token_limit") else record["status"]
            feedback.append({
                "Inputs": {"Problem": record["question"]},
                "Generated Outputs": {"Response": record["output_text"]},
                "Feedback": (
                    f"Score: {score:g}/1. Status: {status}. "
                    f"Parsed answer: {record['parsed_answer']}. Reference answer: {record['expected_answer']}."
                ),
            })
        return {name: feedback for name in components_to_update}


def read_config(path, model=None):
    config = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if model is not None:
        config["model"] = model
    providers.validate_config(config)
    return config


def optimize_prompt(run_dir, spec, train, val, checkpoint, deadline):
    from gepa import optimize
    from gepa.logging.logger import Logger
    from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal, StatelessReflectionLM
    from gepa.strategies.eval_policy import FullEvaluationPolicy

    state_dir = run_dir / "optimizer"
    state_dir.mkdir(exist_ok=True)
    with ExitStack() as stack:
        config = spec["task"]
        reflection_config = spec["reflection"]
        task_client = stack.enter_context(providers.create_client(config))
        reflection_client = stack.enter_context(providers.create_client(reflection_config))
        parallelism = spec["parallelism"]
        adapter = OptiMindAdapter(task_client, config, run_dir, spec["max_metric_calls"], parallelism, checkpoint)
        reflection_error = None
        paused = False
        validation_time = ceil(len(val) / parallelism) * config["timeout_seconds"]
        iteration_time = (
            2 * ceil(len(DIFFICULTIES) / parallelism) * config["timeout_seconds"]
            + validation_time + reflection_config["timeout_seconds"]
        )
        has_state = (state_dir / "gepa_state.bin").is_file()
        if not has_state and deadline is not None and monotonic() + validation_time > deadline:
            return evaluate.PAUSED

        def reflection_lm(prompt):
            try:
                text, usage = providers.reflect(reflection_client, reflection_config, prompt)
            except RuntimeError as exc:
                status = "limit" if isinstance(exc, providers.ReflectionLimit) else "error"
                append_log(run_dir / "reflections.jsonl", {"input": prompt, "status": status, "error": str(exc)})
                raise
            append_log(run_dir / "reflections.jsonl", {"input": prompt, "output": text, "usage": usage, "status": "completed"})
            return text

        class LimitedReflection:
            def __init__(self):
                self.reflector = StatelessReflectionLM(reflection_lm)

            def reflect(self, candidate, dataset, components):
                nonlocal reflection_error
                try:
                    proposal, _ = self.reflector.reflect(candidate, dataset, components)
                except providers.ReflectionLimit:
                    proposal = ReflectionProposal(new_texts={})
                except RuntimeError as exc:
                    reflection_error = exc
                    proposal = ReflectionProposal(new_texts={})
                return proposal, self

        class SavedValidationPolicy(FullEvaluationPolicy):
            def get_seed_eval_batch(self, loader):
                # GEPA 0.1.4 asks for the seed evaluation before loading state.
                return [] if has_state else loader.all_ids()

        class PersistState:
            def on_state_saved(self, event):
                checkpoint()

        # Reserve a full iteration before starting so a promoted candidate can
        # always receive its complete validation evaluation within the budget.
        iteration_budget = 2 * len(DIFFICULTIES) + len(val)

        def should_stop(state):
            nonlocal paused
            if reflection_error is not None or adapter.calls + iteration_budget > spec["max_metric_calls"]:
                return True
            paused = deadline is not None and monotonic() + iteration_time > deadline
            return paused

        result = optimize(
            seed_candidate={COMPONENT: spec["seed_prompt"]},
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_strategy=LimitedReflection(),
            batch_sampler=DifficultySampler(train, spec["seed"]),
            skip_perfect_score=True,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            acceptance_criterion="strict_improvement",
            val_evaluation_policy=SavedValidationPolicy(),
            use_merge=False,
            max_metric_calls=spec["max_metric_calls"],
            stop_callbacks=should_stop,
            run_dir=str(state_dir),
            logger=Logger(str(state_dir / "run_log.txt")),
            callbacks=[PersistState()] if checkpoint else None,
            seed=spec["seed"],
        )
        if reflection_error:
            raise reflection_error
        if paused:
            if checkpoint:
                checkpoint()
            return evaluate.PAUSED
        result_data = result.to_dict()
        result_data["task_api_calls_logged"] = adapter.calls
        (run_dir / "optimization.json").write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")
        prompt = run_dir / "best_prompt.txt"
        temporary = prompt.with_suffix(".tmp")
        temporary.write_text(result.best_candidate[COMPONENT] + "\n", encoding="utf-8")
        temporary.replace(prompt)
        if checkpoint:
            checkpoint()


def main(argv=None, checkpoint=None, deadline=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="OptiMind Baseline task config")
    parser.add_argument("--model", help="Override the task model ID within its provider")
    parser.add_argument("--reflection-config", help="Optional separate model/provider config for reflection")
    parser.add_argument("--reflection-model", help="Override the reflection model ID within its provider")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-metric-calls", type=int, default=DEFAULT_METRIC_CALLS, help="Train and validation rollout budget")
    parser.add_argument("--parallelism", type=int, default=DEFAULT_PARALLELISM, help="Concurrent requests within one batch")
    parser.add_argument("--run-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = read_config(args.config, args.model)
        reflection_model = args.reflection_model or (args.model if not args.reflection_config else None)
        reflection_config = read_config(args.reflection_config or args.config, reflection_model)
        reflection_config.setdefault("provider", "openai")
        reflection_config.update(timeout_seconds=REFLECTION_TIMEOUT, max_output_tokens=REFLECTION_MAX_TOKENS)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if config.get("dataset") != "industryor_optimind":
        parser.error("Select an OptiMind task config")
    if "provider" not in config or args.parallelism < 1:
        parser.error("GEPA requires an explicit provider and positive parallelism")
    rows, splits, split_manifest = load_experiment_dataset("industryor_optimind")
    train = [dict(row, split="train") for row in rows if row["index"] in splits["train"]]
    val = [dict(row, split="val") for row in rows if row["index"] in splits["val"]]
    minimum_budget = 2 * len(val) + 2 * len(DIFFICULTIES)
    if args.max_metric_calls < minimum_budget:
        parser.error(f"Use at least {minimum_budget} metric calls for the seed validation and one full iteration")
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "api_calls": 0, "task_model": config["model"],
            "task_provider": config.get("provider", "openai"), "reflection_model": reflection_config["model"],
            "reflection_provider": reflection_config.get("provider", "openai"), "counts": split_manifest["counts"],
            "difficulty_counts": split_manifest["difficulty_counts"],
            "train_batch": list(DIFFICULTIES), "max_metric_calls": args.max_metric_calls,
            "parallelism": args.parallelism, "task_timeout_seconds": config["timeout_seconds"],
            "task_max_output_tokens": config["max_output_tokens"],
            "reflection_timeout_seconds": REFLECTION_TIMEOUT, "reflection_max_output_tokens": REFLECTION_MAX_TOKENS,
            "final_test_calls": len(splits["test"]),
        }, indent=2))
        return 0
    if not args.run_dir:
        parser.error("--run-dir is required for a paid run")
    run_dir = (ROOT / args.run_dir).resolve()
    spec = {
        "task": config, "reflection": reflection_config, "seed": args.seed,
        "max_metric_calls": args.max_metric_calls, "parallelism": args.parallelism, "split_manifest": split_manifest,
        "split_ids": splits,
        "seed_prompt": evaluation.optimind_solving_instructions(
            (ROOT / config["prompt_file"]).read_text(encoding="utf-8")
        ),
        "code_sha256": {Path(module.__file__).name: digest(module.__file__) for module in (providers, evaluation)},
        "optimizer_sha256": digest(__file__),
    }
    manifest = run_dir / "manifest.json"
    if args.resume:
        if json.loads(manifest.read_text(encoding="utf-8")) != spec:
            parser.error("Resume requires the original model, prompt, budget, split, and evaluator")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    load_environment()
    best_prompt = run_dir / "best_prompt.txt"
    if not best_prompt.exists():
        try:
            if optimize_prompt(run_dir, spec, train, val, checkpoint, deadline) == evaluate.PAUSED:
                return evaluate.PAUSED
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))

    test_dir = run_dir / "test"
    test_args = ["--config", args.config, "--model", config["model"], "--split", "test",
                 "--prompt-file", str(best_prompt), "--run-dir", str(test_dir)]
    if test_dir.exists():
        test_args.append("--resume")
    return evaluate.main(test_args, checkpoint=checkpoint, deadline=deadline)


if __name__ == "__main__":
    sys.exit(main())
