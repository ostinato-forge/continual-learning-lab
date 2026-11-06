import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from continual_lab import (
    ReplayMemory, accuracy, fit_method, make_model, make_world, model_digest,
    pack, training_batch, validate_config,
)
from run_experiment import decision


ROOT = Path(__file__).resolve().parents[1]


def small_config():
    c = json.loads((ROOT / "configs/pilot.json").read_text())
    c.update(
        seeds=[3], tasks=3, features=4, train_examples=32,
        validation_examples=16, test_examples=32, hidden_width=12,
        steps_per_task=3, batch_size=8, memory_capacity=6,
        curve_steps=[0, 1, 3],
    )
    c["decision"]["required_seed_wins"] = 1
    return c


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.c = small_config()
        self.world = make_world(self.c, 3)

    def test_split_ids_and_inputs_are_disjoint(self):
        ids = []
        rows = []
        for split in ("train", "validation", "test"):
            examples = getattr(self.world, split)
            ids.append(set(np.concatenate([e.ids for e in examples]).tolist()))
            rows.append({row[:self.c["features"]].numpy().tobytes() for e in examples for row in e.x})
        for i in range(3):
            for j in range(i):
                self.assertFalse(ids[i] & ids[j])
                self.assertFalse(rows[i] & rows[j])

    def test_world_is_reproducible(self):
        again = make_world(self.c, 3)
        np.testing.assert_array_equal(self.world.order, again.order)
        for first, second in zip(self.world.train, again.train):
            self.assertTrue(torch.equal(first.x, second.x))
            self.assertTrue(torch.equal(first.y, second.y))
        changed = make_world(self.c, 4)
        self.assertFalse(torch.equal(self.world.train[0].x, changed.train[0].x))

    def test_labels_follow_declared_rule(self):
        for task, examples in enumerate(self.world.test):
            raw = examples.x.numpy()[:, :self.c["features"]]
            scores = raw @ self.world.weights[task].T
            bins = (scores[:, 0] > 0).astype(int) + 2 * (scores[:, 1] > 0)
            expected = self.world.label_maps[task][bins]
            np.testing.assert_array_equal(examples.y.numpy(), expected)
            self.assertTrue(torch.all(examples.x[:, self.c["features"] + task] == 1))

    def test_evaluation_does_not_update_weights(self):
        model = make_model(self.c, 3)
        before = model_digest(model)
        score = accuracy(model, self.world.test[0])
        self.assertEqual(before, model_digest(model))
        self.assertTrue(0 <= score <= 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_memory_is_bounded_and_rejects_evaluation_data(self):
        memory = ReplayMemory(6, 7, 3)
        with self.assertRaises(ValueError):
            memory.add(self.world.test[0])
        seen = set()
        for examples in self.world.train:
            memory.add(examples)
            seen.update(examples.ids.tolist())
            self.assertLessEqual(memory.size, 6)
            self.assertTrue(set(memory.examples().ids.tolist()) <= seen)
            self.assertEqual(len(set(memory.examples().ids.tolist())), memory.size)
        self.assertEqual(memory.seen, 96)

    def test_zero_memory_disables_replay(self):
        memory = ReplayMemory(0, 7, 3)
        memory.add(self.world.train[0])
        self.assertEqual(memory.size, 0)
        batch = training_batch(
            "replay", self.world.train[1], self.world.train[1],
            memory, self.c, np.random.default_rng(1),
        )
        self.assertEqual(len(batch[2]), self.c["batch_size"])
        self.assertEqual(set(batch[3].tolist()), {1})

    def test_training_pools_reject_evaluation_data(self):
        with self.assertRaises(ValueError):
            pack([self.world.train[0], self.world.validation[0]])

    def test_methods_match_budgets_and_initialization(self):
        results = []
        with tempfile.TemporaryDirectory() as temp:
            for method in self.c["methods"]:
                results.append(fit_method(
                    self.c, 3, method, self.world, Path(temp) / method,
                    lambda *args: None, time.perf_counter() + 30,
                ))
            self.assertEqual(len({r["initial_model_sha256"] for r in results}), 1)
            self.assertEqual({r["updates"] for r in results}, {9})
            self.assertEqual({r["update_examples"] for r in results}, {72})
            first_scores = [r["stages"][0]["test_accuracy_by_task"] for r in results]
            self.assertEqual(first_scores[0], first_scores[1])
            self.assertEqual(first_scores[1], first_scores[2])
            self.assertEqual(results[0]["retained_examples"], 96)
            self.assertEqual(results[1]["retained_examples"], 0)
            self.assertEqual(results[2]["retained_examples"], 6)

    def test_bad_config_fails_before_training(self):
        validate_config(self.c)
        for key, value in (
            ("memory_capacity", -1), ("replay_fraction", 1),
            ("seeds", [3, 3]), ("curve_steps", [0, 4]),
            ("methods", ["unknown"]), ("learning_rate", 0),
        ):
            altered = copy.deepcopy(self.c)
            altered[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_config(altered)

    def test_decision_requires_retention_and_acquisition(self):
        rows = [
            {"seed": 3, "method": "online", "old_task_retention_drop": 0.3, "mean_acquisition_accuracy": 0.9},
            {"seed": 3, "method": "replay", "old_task_retention_drop": 0.1, "mean_acquisition_accuracy": 0.88},
        ]
        self.assertEqual(decision(rows, self.c)["passing_seeds"], 1)
        rows[1]["mean_acquisition_accuracy"] = 0.7
        self.assertEqual(decision(rows, self.c)["passing_seeds"], 0)
        rows[1]["mean_acquisition_accuracy"] = 0.9
        rows[1]["old_task_retention_drop"] = 0.29
        self.assertEqual(decision(rows, self.c)["passing_seeds"], 0)


if __name__ == "__main__":
    unittest.main()
