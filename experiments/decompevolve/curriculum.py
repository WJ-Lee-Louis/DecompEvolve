"""GEPA training adapter and train-only curriculum, replayable from run artifacts."""
import json

from experiments.gepa.optimize import DifficultySampler, OptiMindAdapter, append_log
from experiments.decompevolve.diagnosis import PRACTICE_SKILLS, diagnose


def read_log(path):
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            yield from (json.loads(line) for line in handle)


class Curriculum(DifficultySampler):
    def __init__(self, train, seed, run_dir, loader, author):
        super().__init__(train, seed)
        self.loader = loader
        self.author = author
        self.questions = {r["index"]: r.get("question", "") for r in train}
        self.diagnostics = run_dir / "diagnoses.jsonl"
        self.probes = run_dir / "synthetic.jsonl"
        self.results = run_dir / "train_batches.jsonl"
        self.original_only = run_dir / "original_batches.jsonl"
        self.parent_pending = False
        self.base_count = len(train)
        self.history = list(read_log(self.probes))
        loader.add_items([entry["row"] for entry in self.history])

    def next_minibatch_ids(self, loader, state):
        ids = super().next_minibatch_ids(loader, state)
        self.parent_pending = True
        if self.author is None:
            return ids
        iteration = state.i
        if any(r["iteration"] == iteration for r in read_log(self.original_only)):
            return ids
        position = next((pos for pos, entry in enumerate(self.history)
                         if entry["iteration"] == iteration), None)
        if position is None:
            position = self.new_probe(iteration)
        if position is not None:
            ids[iteration % len(ids)] = self.base_count + position
        else:
            append_log(self.original_only, dict(iteration=iteration))
        return ids

    def new_probe(self, iteration):
        targets = {}
        diagnoses = {}
        for entry in read_log(self.diagnostics):
            key = entry["index"], entry["diagnosis"]["stage"]
            if key[0] > 0 and entry["diagnosis"].get("practice_skill") == PRACTICE_SKILLS.get(key[1], ""):
                diagnoses[key] = entry["diagnosis"]
                targets[key] = targets.get(key, 0) + 1
        scores = {i: score for batch in read_log(self.results)
                  for i, score in zip(batch["indices"], batch["scores"]) if i < 0}
        seen = {entry["row"]["question"] for entry in self.history}
        eligible = []
        for (source, stage), frequency in targets.items():
            prior = [p for p in self.history if p["source"] == source and p["stage"] == stage]
            level = 2
            if prior:
                last = prior[-1]
                result = scores.get(last["row"]["index"])
                if result is None:
                    continue
                level = last["level"] + (1 if result else -1)
            if 1 <= level <= 3 and len(prior) < 3:
                eligible.append((len(prior), -frequency, source, stage, level))
        for _, _, source, stage, level in sorted(eligible):
            key = f"{self.seed}:{source}:{stage}:{iteration}"
            fingerprints = {p["certificate"]["fingerprint"] for p in self.history}
            problem = self.author.generate(diagnoses[source, stage], self.questions[source], level, key, fingerprints)
            if problem is None or problem["question"] in seen:
                return None
            position = len(self.history)
            row = dict(index=-1-position, split="train", difficulty=("Easy", "Medium", "Hard")[level-1],
                       question=problem["question"], answer=problem["answer"])
            entry = dict(iteration=iteration, source=source, stage=stage, level=level, skill=PRACTICE_SKILLS[stage],
                         row=row, certificate={k: v for k, v in problem.items() if k not in row})
            append_log(self.probes, entry)
            self.history.append(entry)
            self.loader.add_items([row])
            return position
        return None


class DiagnosticAdapter(OptiMindAdapter):
    def __init__(self, *args, run_dir, ask, sampler, **kwargs):
        super().__init__(*args, run_dir=run_dir, **kwargs)
        self.sampler = sampler
        self.ask = ask
        self.diagnostics = run_dir / "diagnoses.jsonl"
        self.batches = run_dir / "train_batches.jsonl"

    def evaluate(self, batch, candidate, capture_traces=False):
        result = super().evaluate(batch, candidate, capture_traces)
        if self.sampler.parent_pending and batch and all(row["split"] == "train" for row in batch):
            self.sampler.parent_pending = False
            append_log(self.batches, dict(indices=[r["index"] for r in batch], scores=result.scores,
                                          perfect=all(result.scores)))
        return result

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        from experiments.api_baseline.providers import ReflectionLimit

        dataset = super().make_reflective_dataset(candidate, eval_batch, components_to_update)
        for pos, record in enumerate(eval_batch.trajectories):
            if record["correct"]:
                continue
            try:
                item = diagnose(record, self.ask)
            except ReflectionLimit:
                item = dict(stage="unknown", reason="Diagnostic request exceeded limit")
            append_log(self.diagnostics, dict(index=record["index"], diagnosis=item))
            # The base adapter shares this feedback list across components.
            if dataset:
                next(iter(dataset.values()))[pos]["Feedback"] += (
                    "\nTrain diagnosis: " + json.dumps(item, ensure_ascii=False) +
                    "\nUse supported checks as general solving instructions. Preserve successful strategies. "
                    "Avoid copying this instance or its answer. Leave uncertain causes unassigned."
                )
        return dataset
