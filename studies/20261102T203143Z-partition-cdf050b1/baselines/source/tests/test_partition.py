import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from analyze_partition import decisions
from continual_lab import fit_method, make_world, save_world, write_json
from partition_models import architecture_stats, make_partition_model
from partition_study import condition_config, validate_plan
from partition_training import fit_partition
from verify_partition import audit_stage_states


ROOT = Path(__file__).resolve().parents[1]


class PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.base = json.loads((ROOT / "configs/pilot.json").read_text())
        self.plan = json.loads((ROOT / "configs/partition_study.json").read_text())
        self.c = copy.deepcopy(self.base)
        self.c.update(seeds=[3], methods=["online"], tasks=3, features=4,
                      train_examples=32, validation_examples=16, test_examples=32,
                      hidden_width=12, steps_per_task=3, batch_size=8,
                      memory_capacity=0, curve_steps=[0, 1, 3], cpu_threads=1)
        self.c["decision"]["required_seed_wins"] = 1
        self.world = make_world(self.c, 3)

    def fit(self, path, kind, world=None, method="online"):
        architecture = {"kind": kind} if kind == "shared" else {"kind": kind, "widths": [12, 10]}
        return fit_partition(self.c, 3, method, world or self.world, path, lambda *args: None,
                             time.perf_counter() + 60, architecture)

    def test_frozen_parameter_counts_match_actual_models(self):
        validate_plan(self.plan, self.base)
        architectures = [{"kind": "shared"}, *self.plan["architectures"].values()]
        for architecture, expected in zip(architectures, (70660, 70659, 70656)):
            model = make_partition_model(self.base, 3, architecture)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertEqual(architecture_stats(self.base, architecture)["total_parameters"], expected)

    def test_architecture_changes_preserve_training_budget(self):
        configs = [condition_config(self.plan, self.base, p) for p in ("baselines", "heads", "experts")]
        for key in ("seeds", "tasks", "features", "classes", "steps_per_task", "batch_size",
                    "learning_rate", "weight_decay", "train_examples", "test_examples", "cpu_threads"):
            self.assertEqual(configs[0][key], configs[1][key])
            self.assertEqual(configs[0][key], configs[2][key])
        self.assertEqual(configs[1]["memory_capacity"], 0)
        self.assertEqual(configs[2]["memory_capacity"], 0)

    def test_plan_rejects_observed_seeds_and_oversized_architectures(self):
        for mutation in ("seed", "budget", "kind"):
            bad = copy.deepcopy(self.plan)
            if mutation == "seed":
                bad["seeds"][0] = self.plan["excluded_seeds"][0]
            elif mutation == "budget":
                bad["architectures"]["experts"]["widths"] = [256, 256]
            else:
                bad["architectures"]["heads"]["kind"] = "experts"
            with self.assertRaises(ValueError):
                validate_plan(bad, self.base)

    def test_routing_uses_the_supplied_identity_and_rejects_bad_inputs(self):
        for kind in ("heads", "experts"):
            model = make_partition_model(self.c, 3, {"kind": kind, "widths": [12, 10]})
            x = torch.cat([e.x[:2] for e in self.world.train])
            routed = model(x)
            for task in range(self.c["tasks"]):
                group = x[task * 2:task * 2 + 2]
                expected = model.parts[task](model.trunk(group) if kind == "heads" else group[:, :self.c["features"]])
                torch.testing.assert_close(routed[task * 2:task * 2 + 2], expected)
            for identity in (0.0, 1.0, 0.5):
                bad = x.clone()
                bad[:, self.c["features"]:] = identity
                with self.assertRaises(ValueError):
                    model(bad)

    def test_inactive_components_survive_adamw_momentum_and_weight_decay(self):
        for kind in ("heads", "experts"):
            model = make_partition_model(self.c, 3, {"kind": kind, "widths": [12, 10]})
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
            for task in (0, 1, 0, 2):
                before = model.component_hashes()
                optimizer.zero_grad(set_to_none=True)
                examples = self.world.train[task]
                torch.nn.functional.cross_entropy(model(examples.x), examples.y).backward()
                model.check_inactive_gradients(task)
                optimizer.step()
                after = model.component_hashes()
                self.assertNotEqual(before[f"task_{task}"], after[f"task_{task}"])
                for other in range(self.c["tasks"]):
                    if other != task:
                        self.assertEqual(before[f"task_{other}"], after[f"task_{other}"])

    def test_shared_training_path_exactly_matches_the_original(self):
        self.c["memory_capacity"] = 6
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in ("online", "replay", "full_history"):
                original = root / (method + "_original")
                new = root / (method + "_new")
                before = fit_method(self.c, 3, method, self.world, original, lambda *a: None, time.perf_counter() + 60)
                after = self.fit(new, "shared", method=method)
                np.testing.assert_array_equal(np.load(original / "batch_ids.npy"), np.load(new / "batch_ids.npy"))
                first = torch.load(original / "checkpoint.pt", weights_only=True)["model"]
                second = torch.load(new / "checkpoint.pt", weights_only=True)["model"]
                self.assertTrue(all(torch.equal(value, second[key]) for key, value in first.items()))
                for key in ("final_mean_accuracy", "mean_acquisition_accuracy", "old_task_retention_drop", "updates", "update_examples"):
                    self.assertEqual(before[key], after[key])

    def test_current_only_batches_match_across_architectures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = None
            for kind in ("shared", "heads", "experts"):
                self.fit(root / kind, kind)
                draws = np.load(root / kind / "batch_ids.npy")
                if baseline is not None:
                    np.testing.assert_array_equal(draws, baseline)
                baseline = draws

    def test_held_out_labels_do_not_affect_learning(self):
        altered = copy.deepcopy(self.world)
        for examples in altered.validation + altered.test:
            examples.y = (examples.y + 1) % self.c["classes"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ("heads", "experts"):
                normal, changed = root / (kind + "_normal"), root / (kind + "_changed")
                self.fit(normal, kind)
                self.fit(changed, kind, altered)
                self.assertEqual((normal / "component_trace.json").read_text(), (changed / "component_trace.json").read_text())
                np.testing.assert_array_equal(np.load(normal / "batch_ids.npy"), np.load(changed / "batch_ids.npy"))
                first = torch.load(normal / "checkpoint.pt", weights_only=True)["model"]
                second = torch.load(changed / "checkpoint.pt", weights_only=True)["model"]
                self.assertTrue(all(torch.equal(value, second[key]) for key, value in first.items()))

    def test_stage_audit_reproduces_predictions_and_detects_tampering(self):
        for kind in ("heads", "experts"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "data").mkdir()
                (root / "seed_3").mkdir()
                save_world(root / "data/seed_3.npz", self.world)
                folder = root / "seed_3/online"
                result = self.fit(folder, kind)
                architecture = {"kind": kind, "widths": [12, 10]}
                checked = audit_stage_states(root, self.c, architecture)
                self.assertEqual(checked["task_end_states_reproduced"], 3)
                self.assertEqual(checked["inactive_component_checks"], 6)
                if kind == "experts":
                    self.assertEqual(checked["independent_prediction_checks"], 6)
                    self.assertEqual(result["old_task_retention_drop"], 0)
                original = (folder / "metrics.json").read_text()
                bad = copy.deepcopy(result)
                bad["stages"][0]["test_accuracy_by_task"][0] += 0.1
                (folder / "metrics.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(AssertionError, "reproduce reported scores"):
                    audit_stage_states(root, self.c, architecture)
                (folder / "metrics.json").write_text(original)
                states = torch.load(folder / "stage_states.pt", weights_only=True)
                next(iter(states[0]["model"].values())).add_(0.1)
                torch.save(states, folder / "stage_states.pt")
                with self.assertRaisesRegex(AssertionError, "Component hashes"):
                    audit_stage_states(root, self.c, architecture)

    def scored_rows(self):
        return [
            {"seed": seed, "method": method, "final_mean_accuracy": 0.85,
             "mean_acquisition_accuracy": 0.9375 if method == "online" else 0.9,
             "old_task_retention_drop": 0.6 if method == "online" else 0.0}
            for seed in self.plan["seeds"] for method in ("online", "replay", "heads", "experts", "full_history")
        ]

    def test_acceptance_cannot_hide_acquisition_failures_or_rounding(self):
        rows = self.scored_rows()
        for row in rows:
            if row["method"] == "heads":
                row["mean_acquisition_accuracy"] = 0.8874999
            if row["method"] == "experts":
                row["final_mean_accuracy"] = 0.7999999
        outcome = decisions(rows, self.plan)
        self.assertTrue(outcome["replay"]["passed"])
        self.assertFalse(outcome["heads"]["passed"])
        self.assertFalse(outcome["experts"]["passed"])
        self.assertEqual(outcome["heads"]["pairs"][0]["failed_targets"], ["acquisition"])
        self.assertEqual(outcome["experts"]["pairs"][0]["failed_targets"], ["final_accuracy"])

    def test_decision_rejects_missing_and_duplicate_conditions(self):
        rows = self.scored_rows()
        for bad in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(AssertionError):
                decisions(bad, self.plan)


if __name__ == "__main__":
    unittest.main()
