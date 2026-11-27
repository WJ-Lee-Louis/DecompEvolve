"""Create the fixed, difficulty-stratified OptiMind IndustryOR split."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark_support.data import ROOT, digest, group_questions, load_dataset, load_splits

DIRECTORY = ROOT / "data/industryor_optimind"
DATASET = DIRECTORY / "datasets_IndustryOR_OptiMind_int.csv"
DIFFICULTY = DIRECTORY / "difficulty.csv"
SEED = 42
EXCLUDED_ITEMS = {
    39: "The continuous optimum is 36288567.0369458..., while the upstream label is 36288567.",
}
QUOTAS = {
    "train": {"Easy": 16, "Medium": 15, "Hard": 4},
    "val": {"Easy": 9, "Medium": 8, "Hard": 3},
    "test": {"Easy": 11, "Medium": 15, "Hard": 4},
}
REVIEWED_GROUPS = {
    (9, 19): "Toy production with two resources and linking constraints.",
    (16, 49): "Multi-month store purchase and sales with the same inventory limits.",
    (18, 20): "Furniture ordering with the same supplier dependency structure.",
    (25, 50, 51, 82): "Six cyclic demand periods and eight-hour shifts; 25 and 50 share the demand vector.",
    (27, 40): "Three-year investment variants with the same four investment options.",
    (36, 47): "Microcomputer production with the same capacity and profit table.",
    (52, 72): "Family trip selection with minimum group size and logical constraints.",
}


def allocate_strata(groups, difficulty):
    rng = random.Random(SEED)
    remaining = {name: counts.copy() for name, counts in QUOTAS.items()}
    assigned = {name: [] for name in QUOTAS}
    split_sizes = {name: sum(counts.values()) for name, counts in QUOTAS.items()}
    grouped = [group for group in groups if len(group) > 1]
    rng.shuffle(grouped)

    def assign_group(position):
        if position == len(grouped):
            return True
        group = grouped[position]
        required = Counter(difficulty[index] for index in group)
        split_order = list(QUOTAS)
        rng.shuffle(split_order)
        split_order.sort(key=lambda name: len(assigned[name]) / split_sizes[name])
        for name in split_order:
            if any(remaining[name][label] < count for label, count in required.items()):
                continue
            for label, count in required.items():
                remaining[name][label] -= count
            assigned[name].extend(group)
            if assign_group(position + 1):
                return True
            del assigned[name][-len(group):]
            for label, count in required.items():
                remaining[name][label] += count
        return False

    if not assign_group(0):
        raise ValueError("Cannot meet all difficulty quotas without breaking a question group")
    for label in QUOTAS["train"]:
        singles = [group[0] for group in groups if len(group) == 1 and difficulty[group[0]] == label]
        rng.shuffle(singles)
        for name in QUOTAS:
            size = remaining[name][label]
            assigned[name].extend(singles[:size])
            del singles[:size]
    return {name: sorted(indices) for name, indices in assigned.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "splits", help="New output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")

    rows = load_dataset(DATASET)
    with DIFFICULTY.open(encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle, strict=True))
    difficulty = {int(row["index"]): row["difficulty"] for row in labels}
    if len(difficulty) != len(labels):
        raise ValueError("Duplicate difficulty IDs")
    counts = Counter(difficulty[row["index"]] for row in rows)
    if counts != sum((Counter(quota) for quota in QUOTAS.values()), Counter()):
        raise ValueError("Difficulty quotas must cover the integer dataset exactly")

    groups = group_questions(rows, REVIEWED_GROUPS)
    ids = allocate_strata(groups, difficulty)
    actual = {name: dict(Counter(difficulty[index] for index in values)) for name, values in ids.items()}
    if actual != QUOTAS:
        raise ValueError("Unexpected difficulty distribution")
    # Revise the original split after allocation to preserve other assignments.
    ids = {name: [i for i in values if i not in EXCLUDED_ITEMS] for name, values in ids.items()}
    groups = [[i for i in group if i not in EXCLUDED_ITEMS] for group in groups]
    groups = [group for group in groups if group]
    rows = [row for row in rows if row["index"] not in EXCLUDED_ITEMS]
    singletons = {group[0] for group in groups if len(group) == 1}
    candidates = [i for i in ids["train"] if i in singletons and difficulty[i] == "Medium"]
    moved = random.Random(SEED).choice(candidates)
    ids["train"].remove(moved)
    ids["test"] = sorted([*ids["test"], moved])
    manifest = {
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "dataset_sha256": digest(DATASET),
        "difficulty": DIFFICULTY.relative_to(ROOT).as_posix(),
        "difficulty_sha256": digest(DIFFICULTY),
        "seed": SEED,
        "counts": {name: len(values) for name, values in ids.items()},
        "difficulty_counts": {
            name: {label: sum(difficulty[i] == label for i in ids[name]) for label in quota}
            for name, quota in QUOTAS.items()
        },
        "excluded_items": [{"index": i, "reason": reason} for i, reason in EXCLUDED_ITEMS.items()],
        "reassigned_item": {
            "index": moved,
            "from": "train",
            "to": "test",
            "selection": "Seeded random choice among Medium Train items in singleton groups.",
        },
        "group_count": len(groups),
        "grouping": "Repository lexical heuristic plus reviewed variants. Assign multi-item groups to the least filled eligible splits first, then shuffle singletons by difficulty.",
        "protocol": "Within-benchmark prompt optimization. Train supplies feedback, validation ranks prompts, and test is held out.",
        "reviewed_groups": [{"ids": group, "reason": reason} for group, reason in REVIEWED_GROUPS.items()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "splits"
        staged.mkdir()
        outputs = {**{f"{name}_ids": values for name, values in ids.items()}, "groups": groups, "manifest": manifest}
        for name, values in outputs.items():
            (staged / f"{name}.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        load_splits(rows, directory=staged, dataset=DATASET)
        staged.rename(args.output)


if __name__ == "__main__":
    main()
