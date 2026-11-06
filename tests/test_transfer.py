"""Invariants for the frozen shared-feature transfer protocol."""

import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from continual_lab import model_digest
from transfer_lab import METHODS, batch_sequence, decisions, empty_memory, fit_targets, make_model, make_world, prefix, stats, train_source, validate_plan
from transfer_study import ROOT, observed_seeds, require_limits
from verify_transfer import check_stage


def tiny_plan():
    p = json.loads((ROOT / 'configs/transfer_study.json').read_text())
    p.update(seeds=[900019], train_examples=12, validation_examples=8, test_examples=16,
             sample_sizes=[4, 8], candidate_samples=4, reference_samples=8,
             shared_widths=[12, 10], expert_widths=[6, 6], parameter_budget=800,
             max_parameter_difference=300, source_steps=2, target_steps=2,
             batch_size=4, memory_capacity=8, required_successes=1, wall_seconds_limit=120)
    return p


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Load optimizer dependencies before the individual test deadlines.
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])

    def setUp(self):
        self.p = tiny_plan()
        self.seed = self.p['seeds'][0]
        self.world = make_world(self.p, self.seed)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.deadline = time.perf_counter() + 120

    def source(self, kind='shared', name='source'):
        return train_source(self.p, self.seed, self.world, kind, self.root / name, self.deadline)

    def test_protocol_and_parameter_counts(self):
        validate_plan(self.p)
        real = json.loads((ROOT / 'configs/transfer_study.json').read_text())
        validate_plan(real)
        self.assertEqual(stats(real)['total_parameters'], 69960)
        self.assertEqual(stats(real, 'independent')['total_parameters'], 70050)
        for kind in ('shared', 'independent'):
            self.assertEqual(sum(v.numel() for v in make_model(self.p, self.seed, kind).parameters()), stats(self.p, kind)['total_parameters'])

    def test_invalid_plans_and_reserved_seeds_are_rejected(self):
        for key, value in [('seeds', [401]), ('sample_sizes', [8, 4]), ('candidate_samples', 8),
                           ('cpu_threads', 2), ('parameter_budget', 1), ('target_pairs', [[0, 1], [1, 3]])]:
            plan = copy.deepcopy(self.p)
            plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_tasks_share_directions_but_target_pairs_are_new(self):
        source = {tuple(sorted(pair)) for pair in self.p['source_pairs']}
        self.assertTrue(source.isdisjoint({tuple(pair) for pair in self.p['target_pairs']}))
        self.assertTrue(np.allclose(self.world.directions @ self.world.directions.T, np.eye(4), atol=1e-6))
        self.assertEqual(set(self.world.order[:4]), set(range(4)))
        self.assertEqual(set(self.world.order[4:]), {4, 5})
        for task, pair in enumerate(self.p['source_pairs'] + self.p['target_pairs']):
            for split in ('train', 'validation', 'test'):
                data = getattr(self.world, split)[task]
                projected = data.x.numpy() @ self.world.directions[pair].T
                bins = (projected[:, 0] > 0).astype(int) + 2 * (projected[:, 1] > 0)
                self.assertTrue(np.array_equal(self.world.label_maps[task][bins], data.y.numpy()))

    def test_splits_and_nested_prefixes(self):
        ids, rows = set(), set()
        for split in ('train', 'validation', 'test'):
            for data in getattr(self.world, split):
                self.assertTrue(ids.isdisjoint(data.ids.tolist()))
                ids.update(data.ids.tolist())
                for row in data.x.numpy():
                    self.assertNotIn(row.tobytes(), rows)
                    rows.add(row.tobytes())
        self.assertTrue(np.array_equal(prefix(self.world.train[4], 4).ids, prefix(self.world.train[4], 8).ids[:4]))
        with self.assertRaises(ValueError):
            prefix(self.world.test[4], 4)

    def test_frozen_reuse_preserves_source_and_does_not_mutate_checkpoint(self):
        source = self.source()
        before = model_digest(source[0])
        row = fit_targets(self.p, self.seed, self.world, 'reuse', 4, source, self.root / 'reuse', self.deadline)
        self.assertTrue(row['source_predictions_unchanged'])
        self.assertEqual(row['source_accuracy_drop'], 0)
        self.assertEqual(row['initial_model_sha256'], before)
        self.assertEqual(model_digest(source[0]), before)
        self.assertEqual(row['target_cost']['peak_retained_examples'], 0)
        self.assertEqual(row['source_cost']['steps'], 8)
        self.assertEqual(row['target_cost']['steps'], 4)
        self.assertEqual(row['complete_cost']['steps'], 12)

    def test_branches_start_from_the_same_source(self):
        source = self.source()
        a = fit_targets(self.p, self.seed, self.world, 'reuse', 4, source, self.root / 'a', self.deadline)
        b = fit_targets(self.p, self.seed, self.world, 'reuse', 8, source, self.root / 'b', self.deadline)
        replay = fit_targets(self.p, self.seed, self.world, 'replay', 4, source, self.root / 'r', self.deadline)
        self.assertEqual(a['initial_model_sha256'], b['initial_model_sha256'])
        self.assertEqual(a['initial_model_sha256'], replay['initial_model_sha256'])
        self.assertEqual(a['source_cost'], replay['source_cost'])
        self.assertLessEqual(replay['complete_cost']['peak_retained_examples'], self.p['memory_capacity'])

    def test_current_only_methods_use_the_same_target_draws(self):
        sources = {'shared': self.source(), 'independent': self.source('independent', 'experts')}
        expected = {}
        for method in ('reuse', 'random_features', 'fresh', 'independent'):
            folder = self.root / method
            fit_targets(self.p, self.seed, self.world, method, 4,
                        sources['independent' if method == 'independent' else 'shared'], folder, self.deadline)
            for stage in range(2):
                draws = np.load(folder / f'stage_{stage}/batch_ids.npy', allow_pickle=False)
                if stage in expected:
                    self.assertTrue(np.array_equal(draws, expected[stage]))
                else:
                    expected[stage] = draws

    def test_held_out_labels_cannot_change_source_or_target_learning(self):
        source_a = self.source(name='source_a')
        changed = copy.deepcopy(self.world)
        for split in ('validation', 'test'):
            for data in getattr(changed, split):
                data.y = (data.y + 1) % 4
        source_b = train_source(self.p, self.seed, changed, 'shared', self.root / 'source_b', self.deadline)
        self.assertEqual(model_digest(source_a[0]), model_digest(source_b[0]))
        for name, world, source in [('a', self.world, source_a), ('b', changed, source_b)]:
            fit_targets(self.p, self.seed, world, 'reuse', 4, source, self.root / name, self.deadline)
        state_a = torch.load(self.root / 'a/stage_1/model.pt', weights_only=True)
        state_b = torch.load(self.root / 'b/stage_1/model.pt', weights_only=True)
        self.assertTrue(all(torch.equal(state_a[key], state_b[key]) for key in state_a))

    def test_stage_verification_rejects_a_held_out_training_id(self):
        self.source()
        folder = self.root / 'source/stage_0'
        draws = np.load(folder / 'batch_ids.npy', allow_pickle=False)
        draws[0, 0] = self.world.test[0].ids[0]
        np.save(folder / 'batch_ids.npy', draws, allow_pickle=False)
        task = int(self.world.order[0])
        with self.assertRaisesRegex(AssertionError, 'Training draws'):
            check_stage(self.p, self.seed, self.world, folder, make_model(self.p, self.seed),
                        empty_memory(self.p, self.seed, True), self.world.train[task], [task],
                        self.p['source_steps'], True, {})

    def test_scored_and_incomplete_plan_seeds_are_observed(self):
        folder = self.root / 'studies/incomplete'
        folder.mkdir(parents=True)
        (folder / 'plan.json').write_text(json.dumps({'seeds': [101, 102]}))
        self.assertEqual(observed_seeds(self.root), {101, 102})

    def test_launcher_is_required(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, 'limited_run'):
            require_limits()

    def test_primary_decision_uses_every_condition(self):
        rows = []
        for method in METHODS:
            for size in self.p['sample_sizes']:
                accuracy = 0.82 if method == 'fresh' else 0.75 if method == 'random_features' else 0.80
                rows.append({'seed': self.seed, 'method': method, 'sample_size': size,
                             'target_mean_accuracy': accuracy, 'target_test_by_task': {'4': accuracy, '5': accuracy},
                             'source_predictions_unchanged': True})
        self.assertTrue(decisions(self.p, rows)['passed'])
        candidate = next(row for row in rows if row['method'] == 'reuse' and row['sample_size'] == 4)
        candidate['target_test_by_task']['4'] = 0.79999999
        verdict = decisions(self.p, rows)
        self.assertFalse(verdict['passed'])
        self.assertIn('each_target_accuracy', verdict['pairs'][0]['failed_targets'])
        with self.assertRaises(AssertionError):
            decisions(self.p, rows[:-1])


if __name__ == '__main__':
    unittest.main()
