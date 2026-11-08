"""Create fixed ID splits without modifying the CSV or its answer labels."""
import argparse
import json
from decompevolve.data import DATASET, SPLITS, allocate_groups, digest, group_questions, load_dataset, load_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Explicitly replace existing splits")
    args = parser.parse_args()
    rows = load_dataset()
    if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= len(rows):
        parser.error("Sizes must leave nonempty train, val, and test sets")
    if any(SPLITS.glob("*.json")) and not args.force:
        parser.error("Splits already exist; use --force only when intentionally replacing them")
    overrides_path = DATASET.parent / "group_overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    groups = group_questions(rows, overrides["groups"])
    ids = allocate_groups(groups, args.val_size, args.test_size, args.seed)
    manifest = {
        "dataset": str(DATASET.relative_to(DATASET.parents[1])),
        "dataset_sha256": digest(DATASET),
        "seed": args.seed,
        "counts": {name: len(values) for name, values in ids.items()},
        "grouping": "question_lexical_v1: same 12-token opening OR 3-gram Jaccard >= .55 and numeric multiset Dice >= .80",
        "group_count": len(groups),
        "reviewed_overrides_sha256": digest(overrides_path),
        "note": "Heuristic groups require review; no guarantee of semantic deduplication. Answers unchanged. No LP/IP stratification claimed.",
    }
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, values in {**{name + "_ids": values for name, values in ids.items()}, "groups": groups, "manifest": manifest}.items():
        (SPLITS / (name + ".json")).write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_splits(rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
