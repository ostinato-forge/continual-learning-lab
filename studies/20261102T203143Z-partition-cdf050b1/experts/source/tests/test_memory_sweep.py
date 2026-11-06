import copy
import json
from pathlib import Path
import unittest

from memory_sweep import condition_config, meets_target, select_capacity, validate_plan


ROOT = Path(__file__).resolve().parents[1]


class MemorySweepTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / "configs/pilot.json").read_text())
        self.plan = json.loads((ROOT / "configs/memory_sweep.json").read_text())

    def cases(self, successful_capacity=256):
        rows = []
        for capacity in self.plan["capacities"]:
            for seed in self.plan["exploration_seeds"]:
                succeeds = capacity >= successful_capacity
                metrics = {
                    "final_mean_accuracy": 0.85 if succeeds else 0.6 if capacity else 0.35,
                    "mean_acquisition_accuracy": 0.9 if capacity else 0.93,
                    "old_task_retention_drop": 0.1 if capacity else 0.6,
                }
                rows.append({
                    "capacity": capacity, "seed": seed,
                    "validation": metrics, "test": copy.deepcopy(metrics),
                })
        return rows

    def test_plan_has_fresh_disjoint_seeds(self):
        validate_plan(self.plan, self.base)
        self.plan["confirmation_seeds"][0] = self.plan["exploration_seeds"][0]
        with self.assertRaises(ValueError):
            validate_plan(self.plan, self.base)

    def test_plan_rejects_pilot_seed_reuse(self):
        self.plan["exploration_seeds"][0] = self.base["seeds"][0]
        with self.assertRaises(ValueError):
            validate_plan(self.plan, self.base)

    def test_configs_keep_model_and_budget_fixed(self):
        before = copy.deepcopy(self.base)
        allowed = {"experiment", "seeds", "methods", "memory_capacity", "decision"}
        for capacity in self.plan["capacities"]:
            c = condition_config(self.plan, self.base, capacity)
            for key in self.base:
                if key not in allowed:
                    self.assertEqual(c[key], self.base[key])
        self.assertEqual(before, self.base)
        confirmed = condition_config(self.plan, self.base, 256, confirmation=True)
        self.assertEqual(confirmed["seeds"], self.plan["confirmation_seeds"])
        self.assertEqual(set(confirmed["methods"]), {"online", "replay", "full_history"})

    def test_selection_uses_validation_not_test_scores(self):
        rows = self.cases(256)
        first = select_capacity(rows, self.plan)
        for row in rows:
            row["test"] = {
                "final_mean_accuracy": 1 if row["capacity"] == 16 else 0,
                "mean_acquisition_accuracy": 1 if row["capacity"] == 16 else 0,
                "old_task_retention_drop": -1 if row["capacity"] == 16 else 1,
            }
        self.assertEqual(select_capacity(rows, self.plan), first)
        self.assertEqual(first["chosen_capacity"], 256)

    def test_no_qualifying_capacity_is_a_valid_outcome(self):
        result = select_capacity(self.cases(4096), self.plan)
        self.assertIsNone(result["chosen_capacity"])
        self.assertFalse(any(row["qualifies"] for row in result["choices"]))

    def test_selection_requires_enough_individual_seed_successes(self):
        rows = self.cases(256)
        for row in rows:
            if row["capacity"] == 256 and row["seed"] in self.plan["exploration_seeds"][:3]:
                row["validation"]["final_mean_accuracy"] = 0.79
        result = select_capacity(rows, self.plan)
        self.assertEqual(result["chosen_capacity"], 512)
        self.assertEqual(next(r["passing_count"] for r in result["choices"] if r["capacity"] == 256), 7)

    def test_retention_gain_does_not_excuse_acquisition_loss(self):
        baseline = {"final_mean_accuracy": 0.4, "mean_acquisition_accuracy": 0.95, "old_task_retention_drop": 0.6}
        candidate = {"final_mean_accuracy": 0.85, "mean_acquisition_accuracy": 0.8, "old_task_retention_drop": 0.05}
        self.assertFalse(meets_target(candidate, baseline, self.plan))
        candidate["mean_acquisition_accuracy"] = 0.93
        self.assertTrue(meets_target(candidate, baseline, self.plan))

    def test_original_replication_criterion_does_not_include_new_accuracy_target(self):
        baseline = {"final_mean_accuracy": 0.4, "mean_acquisition_accuracy": 0.95, "old_task_retention_drop": 0.6}
        candidate = {"final_mean_accuracy": 0.77, "mean_acquisition_accuracy": 0.93, "old_task_retention_drop": 0.1}
        self.assertFalse(meets_target(candidate, baseline, self.plan))
        self.assertTrue(meets_target(candidate, baseline, self.plan, require_final=False))


if __name__ == "__main__":
    unittest.main()
