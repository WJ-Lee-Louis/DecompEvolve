import json
import tempfile
import unittest
from pathlib import Path

from experiments.decompevolve.author import NanoAuthor, verify
from experiments.decompevolve.curriculum import read_log


class AuthorTests(unittest.TestCase):
    def setUp(self):
        self.model = dict(
            names=["regular supply", "certified supply"],
            upper=[12, 12],
            direction="min",
            integer=False,
            value=[2, 5],
            target=1,
            rows=[
                dict(kind="resource", unit="tons", a=[1, 1], sense=">=", rhs=12),
                dict(kind="ratio", a=[1, -2], sense="<=", rhs=0),
            ],
        )

    def test_verified_minimum_and_relaxed_cost(self):
        item = verify(self.model, "constraints")
        self.assertEqual((item["answer"], item["counterfactual_value"]), (36, "24"))
        self.assertEqual(item["witness"], ["8", "4"])
        with self.assertRaises(ValueError):
            verify(dict(self.model, value=[2, 2]), "constraints")

    def test_other_transforms(self):
        model = dict(self.model, direction="max", target=None)
        packing = dict(kind="resource", unit="kg", a=[2, 2], sense="<=", rhs=5)
        item = verify(dict(model, integer=True, rows=[packing], value=[3, 4]), "domain")
        self.assertEqual((item["answer"], item["counterfactual_value"]), (8, "10"))
        item = verify(
            dict(
                model,
                target=0,
                rows=[dict(kind="time", a=[15, 30], sense="<=", rhs=1)],
                value=[1, 1],
            ),
            "units",
        )
        self.assertEqual((item["answer"], item["counterfactual_value"]), (4, "1/15"))
        pack = dict(packing, a=[1, 1], rhs=5)
        item = verify(
            dict(model, rows=[pack], value=[1, 3], price=[5, 4], cost=[4, 1]),
            "objective",
        )
        self.assertEqual((item["answer"], item["counterfactual_value"]), (15, "25"))
        with self.assertRaisesRegex(ValueError, "share"):
            verify(
                dict(model, rows=[pack], value=[3, 1], price=[5, 4], cost=[2, 3]),
                "objective",
            )
        item = verify(
            dict(
                model,
                rows=[dict(pack, a=[1, 0], rhs=2), dict(pack, a=[0, 1], rhs=3)],
                value=[2, 2],
            ),
            "optimality",
        )
        self.assertEqual((item["answer"], item["counterfactual_value"]), (10, "6"))

    def test_one_repair_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            answers = iter(["{}", json.dumps(self.model)])
            author = NanoAuthor(
                Path(d), lambda _: (next(answers), {"total_tokens": 100}), 30000
            )
            args = (
                {"stage": "constraints"},
                "original train context",
                2,
                "42:13:constraints:1",
                set(),
            )
            self.assertEqual(author.generate(*args)["answer"], 36)
            self.assertEqual(author.generate(*args)["answer"], 36)
            self.assertEqual(len(list(read_log(author.path))), 4)
            exhausted = NanoAuthor(Path(d), None, 200)
            self.assertIsNone(exhausted.generate(*args[:3], "next", set()))
