import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_replay_allocation import audit_allocation
from replay_allocation import condition_config, select_fraction, validate_plan


ROOT = Path(__file__).resolve().parents[1]


class ReplayAllocationTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / "configs/pilot.json").read_text())
        self.plan = json.loads((ROOT / "configs/replay_allocation.json").read_text())

    def cases(self, first_success=0.125):
        rows = []
        for fraction in self.plan["replay_fractions"]:
            for seed in self.plan["exploration_seeds"]:
                metrics = {
                    "final_mean_accuracy": 0.85 if fraction >= first_success else 0.4,
                    "mean_acquisition_accuracy": 0.91 if fraction else 0.94,
                    "old_task_retention_drop": 0.1 if fraction else 0.6,
                }
                rows.append({"fraction": fraction, "seed": seed, "validation": metrics, "test": copy.deepcopy(metrics)})
        return rows

    def test_seed_sets_are_fresh_and_disjoint(self):
        validate_plan(self.plan, self.base)
        for old_seed in (self.base["seeds"][0], self.plan["excluded_seeds"][-1], self.plan["confirmation_seeds"][0]):
            plan = copy.deepcopy(self.plan)
            plan["exploration_seeds"][0] = old_seed
            with self.assertRaises(ValueError):
                validate_plan(plan, self.base)

    def test_model_memory_and_total_budget_stay_fixed(self):
        original = copy.deepcopy(self.base)
        allowed = {"experiment", "methods", "seeds", "decision", "replay_fraction", "memory_capacity"}
        for fraction in self.plan["replay_fractions"]:
            c = condition_config(self.plan, self.base, fraction)
            self.assertEqual(c["memory_capacity"], 1024)
            self.assertEqual(c["replay_fraction"], fraction)
            for key in self.base:
                if key not in allowed:
                    self.assertEqual(c[key], self.base[key])
        self.assertEqual(self.base, original)

    def test_replay_shares_use_whole_example_counts(self):
        counts = [int(v * self.base["batch_size"]) for v in self.plan["replay_fractions"][1:]]
        self.assertEqual(counts, [8, 16, 32])
        self.plan["replay_fractions"][1] = 0.1
        with self.assertRaises(ValueError):
            validate_plan(self.plan, self.base)

    def test_selection_ignores_test_scores(self):
        rows = self.cases(0.25)
        before = select_fraction(rows, self.plan)
        for row in rows:
            row["test"] = {"invalid": "Test scores must not be read during selection"}
        self.assertEqual(select_fraction(rows, self.plan), before)
        self.assertEqual(before["chosen_fraction"], 0.25)

    def test_selection_requires_individual_successes_and_chooses_smallest(self):
        rows = self.cases()
        for row in rows:
            if row["fraction"] == 0.125 and row["seed"] in self.plan["exploration_seeds"][:3]:
                row["validation"]["mean_acquisition_accuracy"] = 0.88
        result = select_fraction(rows, self.plan)
        self.assertEqual(result["chosen_fraction"], 0.25)
        self.assertEqual(result["choices"][0]["passing_count"], 7)

    def test_negative_outcome_is_valid(self):
        result = select_fraction(self.cases(1), self.plan)
        self.assertIsNone(result["chosen_fraction"])
        self.assertFalse(any(row["qualifies"] for row in result["choices"]))

    def test_selection_rejects_missing_or_duplicate_pairs(self):
        for rows in (self.cases()[1:], self.cases() + [self.cases()[0]], self.cases()[:-1]):
            with self.assertRaises(ValueError):
                select_fraction(rows, self.plan)

    def test_confirmation_and_reference_use_reserved_seeds(self):
        confirmation = condition_config(self.plan, self.base, 0.125, "confirmation")
        reference = condition_config(self.plan, self.base, 0.5, "reference")
        for c in (confirmation, reference):
            self.assertEqual(c["seeds"], self.plan["confirmation_seeds"])
            self.assertEqual(c["memory_capacity"], 1024)
        self.assertEqual(confirmation["methods"], ["full_history", "online", "replay"])
        self.assertEqual(reference["methods"], ["replay"])

    def test_batch_audit_detects_wrong_replay_share_and_first_task_replay(self):
        c = {"seeds": [1], "methods": ["replay"], "steps_per_task": 2, "batch_size": 4, "replay_fraction": 0.25}
        correct = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 0], [5, 6, 7, 1]])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "data").mkdir()
            folder = run / "seed_1" / "replay"
            folder.mkdir(parents=True)
            np.savez(run / "data" / "seed_1.npz", order=np.array([0, 1]), train_0_ids=np.arange(4), train_1_ids=np.arange(4, 8))
            np.save(folder / "batch_ids.npy", correct)
            self.assertEqual(audit_allocation(run, c), 4)
            for position, replacement in (((2, 2), 2), ((0, 0), 4)):
                bad = correct.copy()
                bad[position] = replacement
                np.save(folder / "batch_ids.npy", bad)
                with self.assertRaises(AssertionError):
                    audit_allocation(run, c)


if __name__ == "__main__":
    unittest.main()
