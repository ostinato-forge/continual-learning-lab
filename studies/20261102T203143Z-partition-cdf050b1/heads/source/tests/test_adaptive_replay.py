import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from adaptive_controller import AdaptiveController, next_fraction, validate_policy
from adaptive_study import condition_config, validate_plan
from adaptive_training import fit_with_policy
from analyze_adaptive import audit_controller, decisions
from continual_lab import fit_method, make_model, make_world, model_digest, save_world, write_json


ROOT = Path(__file__).resolve().parents[1]


class AdaptiveReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.base = json.loads((ROOT / "configs/pilot.json").read_text())
        self.plan = json.loads((ROOT / "configs/adaptive_replay.json").read_text())
        self.policy = copy.deepcopy(self.plan["controller"])
        self.c = copy.deepcopy(self.base)
        self.c.update(seeds=[3], methods=["replay"], tasks=3, features=4, train_examples=32, validation_examples=16, test_examples=32, hidden_width=12, steps_per_task=3, batch_size=8, memory_capacity=6, curve_steps=[0, 1, 3])
        self.c["decision"]["required_seed_wins"] = 1
        self.policy["probe_interval"] = 1
        self.world = make_world(self.c, 3)

    def fit(self, folder, policy, world=None):
        return fit_with_policy(self.c, 3, "replay", world or self.world, folder, lambda *args: None, time.perf_counter() + 60, policy=policy)

    def test_frozen_plan_uses_fresh_seeds_and_equal_gradient_budgets(self):
        validate_plan(self.plan, self.base)
        fixed = condition_config(self.plan, self.base, "baselines")
        adaptive = condition_config(self.plan, self.base, "adaptive")
        for key in self.base:
            if key not in ("experiment", "methods", "replay_fraction"):
                self.assertEqual(fixed[key], adaptive[key])
        self.plan["seeds"][0] = self.plan["excluded_seeds"][-1]
        with self.assertRaises(ValueError):
            validate_plan(self.plan, self.base)

    def test_policy_rejects_fractional_counts_and_crossed_thresholds(self):
        validate_policy(self.policy, self.c)
        invalid = copy.deepcopy(self.policy)
        invalid["fractions"] = [0.1, 0.25, 0.5, 0.75]
        with self.assertRaises(ValueError):
            validate_policy(invalid, self.c)
        invalid = copy.deepcopy(self.policy)
        invalid["decrease_drop"] = 0.1
        with self.assertRaises(ValueError):
            validate_policy(invalid, self.c)

    def test_controller_increases_on_forgetting_and_decreases_on_recovery(self):
        self.assertEqual(next_fraction(0.25, 1.0, 0.9375, self.policy), 0.5)
        self.assertEqual(next_fraction(0.5, 1.0, 0.96875, self.policy), 0.5)
        self.assertEqual(next_fraction(0.5, 1.0, 1.0, self.policy), 0.25)
        self.assertEqual(next_fraction(0.75, 1.0, 0.5, self.policy), 0.75)
        self.assertEqual(next_fraction(0.125, 1.0, 1.0, self.policy), 0.125)

    def test_probe_rejects_held_out_data_and_does_not_train_model(self):
        controller = AdaptiveController(self.policy, 3)
        with self.assertRaises(ValueError):
            controller.begin_stage(self.world.validation[0], 1)
        controller.begin_stage(self.world.train[0], 1)
        model = make_model(self.c, 3)
        before = model_digest(model)
        self.assertEqual(controller.before_update(model, 0), 0.25)
        self.assertEqual(model_digest(model), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(len(set(controller.trace[0]["ids"])), len(controller.trace[0]["ids"]))
        controller.fraction = 0.75
        controller.begin_stage(self.world.train[1], 2)
        self.assertEqual(controller.fraction, 0.25)
        self.assertIsNone(controller.reference)

    def test_fixed_policy_path_exactly_matches_original_training(self):
        constant = {**self.policy, "fractions": [0.5], "initial_fraction": 0.5}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = fit_method(self.c, 3, "replay", self.world, root / "original", lambda *args: None, time.perf_counter() + 60)
            for name, policy in (("disabled", None), ("constant", constant)):
                actual = self.fit(root / name, policy)
                np.testing.assert_array_equal(np.load(root / "original/batch_ids.npy"), np.load(root / name / "batch_ids.npy"))
                expected_weights = torch.load(root / "original/checkpoint.pt", weights_only=True)["model"]
                actual_weights = torch.load(root / name / "checkpoint.pt", weights_only=True)["model"]
                self.assertTrue(all(torch.equal(value, actual_weights[key]) for key, value in expected_weights.items()))
                for key in ("final_mean_accuracy", "mean_acquisition_accuracy", "old_task_retention_drop", "updates", "update_examples", "initial_model_sha256"):
                    self.assertEqual(original[key], actual[key])

    def test_held_out_labels_cannot_change_learning_or_controller(self):
        altered = copy.deepcopy(self.world)
        for examples in altered.validation + altered.test:
            examples.y = (examples.y + 1) % 4
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fit(root / "normal", self.policy)
            self.fit(root / "changed", self.policy, altered)
            for name in ("batch_ids.npy", "replay_counts.npy"):
                np.testing.assert_array_equal(np.load(root / "normal" / name), np.load(root / "changed" / name))
            self.assertEqual((root / "normal/controller_trace.json").read_text(), (root / "changed/controller_trace.json").read_text())
            first = torch.load(root / "normal/checkpoint.pt", weights_only=True)["model"]
            second = torch.load(root / "changed/checkpoint.pt", weights_only=True)["model"]
            self.assertTrue(all(torch.equal(value, second[key]) for key, value in first.items()))

    def test_audit_reproduces_probes_and_detects_bad_access_decisions_and_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "seed_3").mkdir()
            save_world(root / "data/seed_3.npz", self.world)
            folder = root / "seed_3/replay"
            result = self.fit(folder, self.policy)
            write_json(root / "summary.json", {"results": [result]})
            checked = audit_controller(root, self.c, self.policy)
            self.assertEqual(checked["probe_states_reproduced"], 6)
            self.assertEqual(checked["probe_forward_examples_checked"], 36)
            trace = json.loads((folder / "controller_trace.json").read_text())
            for field, value in (("ids", self.world.test[0].ids[:6].tolist()), ("fraction_after", 0.75)):
                bad = copy.deepcopy(trace)
                bad[0][field] = value
                (folder / "controller_trace.json").write_text(json.dumps(bad))
                with self.assertRaises(AssertionError):
                    audit_controller(root, self.c, self.policy)
            (folder / "controller_trace.json").write_text(json.dumps(trace))
            counts = np.load(folder / "replay_counts.npy")
            self.assertTrue(np.all(counts[:self.c["steps_per_task"]] == 0))
            counts[-1] += 1
            np.save(folder / "replay_counts.npy", counts)
            with self.assertRaises(AssertionError):
                audit_controller(root, self.c, self.policy)

    def test_good_retention_cannot_hide_failed_acquisition(self):
        rows = []
        for seed in self.plan["seeds"]:
            rows.extend([
                {"seed": seed, "method": "online", "final_mean_accuracy": 0.4, "mean_acquisition_accuracy": 0.94, "old_task_retention_drop": 0.6},
                {"seed": seed, "method": "replay", "final_mean_accuracy": 0.85, "mean_acquisition_accuracy": 0.9, "old_task_retention_drop": 0.1},
                {"seed": seed, "method": "adaptive", "final_mean_accuracy": 0.9, "mean_acquisition_accuracy": 0.86, "old_task_retention_drop": 0.0},
            ])
        result = decisions(rows, self.plan)
        self.assertTrue(result["replay"]["passed"])
        self.assertFalse(result["adaptive"]["passed"])
        self.assertEqual(result["adaptive"]["passing_count"], 0)


if __name__ == "__main__":
    unittest.main()
