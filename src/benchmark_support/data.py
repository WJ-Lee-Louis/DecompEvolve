"""Strict CSV loading and deterministic, question-only grouping."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/datasets_ICML_int.csv"
SPLITS = ROOT / "data/splits"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path=DATASET):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != ["index", "question", "answer"]:
            raise ValueError("Expected CSV columns: index, question, answer")
        rows = []
        for row in reader:
            if None in row or any(row[k] is None for k in reader.fieldnames):
                raise ValueError("Malformed CSV record")
            if not row["question"].strip():
                raise ValueError("Empty question")
            if not re.fullmatch(r"[+-]?[0-9]+", row["answer"]):
                raise ValueError("Noninteger answer at index " + row["index"])
            rows.append(dict(index=int(row["index"]), question=row["question"], answer=int(row["answer"])))
    if len({r["index"] for r in rows}) != len(rows):
        raise ValueError("Duplicate indices")
    return rows


def group_questions(rows, overrides=()):
    """Conservative lexical heuristic, not a guarantee of semantic deduplication.

    Merge shared 12-token openings or high 3-gram overlap with similar numeric
    content. Labels are never used. Keep the detected groups reviewable on disk.
    """
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    features = []
    for row in rows:
        text = row["question"].lower().replace(",", "")
        tokens = re.findall(r"[a-z]+|[0-9]+(?:\.[0-9]+)?", text)
        grams = set(zip(tokens, tokens[1:], tokens[2:]))
        numbers = Counter(re.findall(r"[0-9]+(?:\.[0-9]+)?", text))
        features.append((tokens, grams, numbers))
    for i, (ta, ga, na) in enumerate(features):
        for j in range(i):
            tb, gb, nb = features[j]
            same_opening = len(ta) >= 12 and len(tb) >= 12 and ta[:12] == tb[:12]
            overlap = len(ga & gb) / max(1, len(ga | gb))
            numeric = 2 * sum((na & nb).values()) / max(1, sum(na.values()) + sum(nb.values()))
            if same_opening or (overlap >= 0.55 and numeric >= 0.80):
                parents[find(i)] = find(j)
    positions = {row["index"]: i for i, row in enumerate(rows)}
    for group in overrides:
        if len(group) < 2 or any(i not in positions for i in group):
            raise ValueError("Invalid reviewed group override")
        for index in group[1:]:
            parents[find(positions[index])] = find(positions[group[0]])
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row["index"])
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def allocate_groups(groups, val_size, test_size, seed):
    """Exact-size allocation via subset dynamic programming, with seeded order."""
    shuffled = [list(g) for g in groups]
    random.Random(seed).shuffle(shuffled)
    states = {(0, 0): None}
    for group in shuffled:
        additions = {}
        for (v, t), path in list(states.items()):
            for split, target in (("val", (v + len(group), t)), ("test", (v, t + len(group)))):
                if target[0] <= val_size and target[1] <= test_size and target not in states and target not in additions:
                    additions[target] = (path, split, group)
        states.update(additions)
        if (val_size, test_size) in states:
            break
    if (val_size, test_size) not in states:
        raise ValueError("Cannot reach requested sizes without breaking groups")
    assigned = {"train": [], "val": [], "test": []}
    path = states[(val_size, test_size)]
    selected = set()
    while path is not None:
        previous, split, group = path
        assigned[split].extend(group)
        selected.update(group)
        path = previous
    assigned["train"] = [i for group in groups for i in group if i not in selected]
    return {key: sorted(ids) for key, ids in assigned.items()}


def load_splits(rows, directory=SPLITS, dataset=DATASET):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["dataset_sha256"] != digest(dataset):
        raise ValueError("Dataset changed since split creation")
    ids = {name: json.loads((directory / (name + "_ids.json")).read_text()) for name in ("train", "val", "test")}
    flat = [i for values in ids.values() for i in values]
    if len(flat) != len(set(flat)) or set(flat) != {r["index"] for r in rows}:
        raise ValueError("Split IDs must cover the dataset exactly once")
    if any(len(ids[name]) != manifest["counts"][name] for name in ids):
        raise ValueError("Split sizes differ from manifest")
    membership = {i: name for name, values in ids.items() for i in values}
    groups = json.loads((directory / "groups.json").read_text())
    group_flat = [i for group in groups for i in group]
    if len(group_flat) != len(set(group_flat)) or set(group_flat) != set(flat):
        raise ValueError("Invalid group coverage")
    if any(len({membership[i] for i in group}) != 1 for group in groups):
        raise ValueError("A detected question group crosses splits")
    return ids
