from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DataContextTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "fixtures/preference_portfolios.json").read_text(encoding="utf-8"))

    def test_ballots_reference_real_proposals(self):
        proposals = {row["id"] for row in self.data["proposals"]}
        for row in self.data["ballots"]:
            self.assertTrue({key for rank in row["ranking"] for key in rank} <= proposals)
        self.assertTrue(any(len(rank) > 1 for row in self.data["ballots"] for rank in row["ranking"]))

    def test_revisions_absences_and_larger_search_space(self):
        self.assertTrue(any(row["revision"] > 1 for row in self.data["ballots"]))
        self.assertTrue(self.data["absent_representatives"])
        self.assertGreater(len(self.data["larger_case"]["proposals"]), 25)

