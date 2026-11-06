"""Audit frozen inputs, allowed draws, and saved transfer predictions."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from continual_lab import digest, model_digest
from transfer_lab import METHODS, batch_sequence, cost, decisions, empty_memory, evaluate, linear_reference, make_model, make_world, prefix, stats, validate_plan, world_arrays
from transfer_study import ROOT, require_limits


DEPENDENCIES = ('continual_lab.py', 'transfer_lab.py', 'transfer_study.py', 'verify_transfer.py')


def read(path):
    return json.loads(path.read_text())


def verify_hashes(study):
    summary, manifest = read(study / 'summary.json'), read(study / 'manifest.json')
    if summary['status'] != 'completed' or digest(study / 'manifest.json') != summary['manifest_sha256']:
        raise AssertionError('Study is incomplete or the input manifest changed')
    for listing in (manifest, read(study / 'artifacts.json')):
        for name, expected in listing.items():
            if digest(study / name) != expected:
                raise AssertionError('Saved artifact changed: ' + name)
    for name in DEPENDENCIES:
        if digest(ROOT / name) != manifest['source/' + name]:
            raise AssertionError('Use the frozen experiment dependency: ' + name)
    return summary


def check_stage(p, seed, world, folder, model, memory, current, seen, steps, replay, counters):
    row = read(folder / 'stage.json')
    before = model.hashes()
    frozen = model.kind == 'shared' and not next(model.trunk.parameters()).requires_grad
    if (row['initial_components'] != before or row['task'] != int(current.task_ids[0])
            or row['seen_tasks'] != seen or row['steps'] != steps or row['replay'] != replay
            or row['frozen_features'] != frozen):
        raise AssertionError('Stage differs from its planned starting state')
    old_ids = memory.examples().ids.copy()
    for name, expected in (('current_ids', current.ids), ('memory_before_ids', old_ids)):
        if not np.array_equal(np.load(folder / (name + '.npy'), allow_pickle=False), expected):
            raise AssertionError('Permitted examples or memory membership changed')
    expected_batches = np.stack([batch[2] for batch in batch_sequence(p, seed, row['task'], current,
                                                                    memory.examples(), steps, replay)])
    batches = np.load(folder / 'batch_ids.npy', allow_pickle=False)
    if not np.array_equal(batches, expected_batches):
        raise AssertionError('Training draws differ from permitted deterministic sampling')
    if replay:
        memory.add(current)
    if not np.array_equal(np.load(folder / 'memory_after_ids.npy', allow_pickle=False), memory.examples().ids):
        raise AssertionError('Reservoir membership differs from its fixed rule')
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True))
    if model_digest(model) != row['model_sha256'] or model.hashes() != row['final_components']:
        raise AssertionError('Checkpoint component hashes disagree')
    for name in before:
        protected = (name == 'trunk' and frozen) or (name.startswith('task_') and int(name[5:]) not in (set(seen) if replay else {row['task']}))
        if protected:
            counters['protected_components_checked'] += 1
            if before[name] != model.hashes()[name]:
                raise AssertionError('Protected parameters changed')
    if evaluate(model, world, seen) != row['scores']:
        raise AssertionError('Checkpoint scores did not reproduce')
    current_draws = batches[np.isin(batches, current.ids)]
    count_old = int(p['batch_size'] * p['replay_fraction']) if replay and len(old_ids) else 0
    arithmetic = stats(p, model.kind, frozen)
    expected = {
        'current_available': len(current.ids), 'unique_current_draws': len(np.unique(current_draws)),
        'presentations': steps * p['batch_size'],
        'current_presentations': steps * (p['batch_size'] - count_old),
        'replay_presentations': steps * count_old,
        'training_dense_macs': steps * p['batch_size'] * arithmetic['training_dense_macs_per_example'],
        'evaluation_dense_macs': len(seen) * (p['validation_examples'] + p['test_examples']) * arithmetic['forward_dense_macs_per_example'],
        'retained_examples': memory.size if replay else 0,
        'retained_example_bytes': (p['features'] * 4 + 24) * memory.size if replay else 0,
    }
    if any(row[key] != value for key, value in expected.items()):
        raise AssertionError('A stage count disagrees with the evidence')
    if any(row[key] < 0 for key in row if key.endswith('_seconds') or key.endswith('_bytes')):
        raise AssertionError('Negative resource measurement')
    counters['checkpoints_reproduced'] += 1
    counters['training_example_draws_checked'] += batches.size
    return row


def check_decision_consistency(p, summary):
    if decisions(p, summary['results']) != summary['decision']:
        raise AssertionError('Primary decisions disagree with unrounded outcomes')


def verify_study(study):
    summary = verify_hashes(study)
    p = read(study / 'plan.json')
    validate_plan(p)
    audit = read(study / 'seed_audit.json')
    if not audit['unobserved'] or set(audit['previously_observed']) & set(p['seeds']):
        raise AssertionError('Scored seeds were previously observed')
    if audit['purpose'] != summary['purpose']:
        raise AssertionError('Pipeline and scored evidence were mixed')
    counters = {'status': 'passed', 'checkpoints_reproduced': 0, 'training_example_draws_checked': 0,
                'protected_components_checked': 0, 'paired_current_batches_checked': 0,
                'source_prediction_pairs_checked': 0, 'linear_references_reproduced': 0,
                'datasets_regenerated': 0}
    collected, all_references, source_rows = [], [], []
    for seed in p['seeds']:
        world = make_world(p, seed)
        expected_arrays = world_arrays(world)
        with np.load(study / 'data' / f'seed_{seed}.npz', allow_pickle=False) as saved:
            if set(saved.files) != set(expected_arrays) or any(not np.array_equal(saved[name], value) for name, value in expected_arrays.items()):
                raise AssertionError('Dataset differs from the frozen generator')
        counters['datasets_regenerated'] += 1
        sources = {}
        for kind in ('shared', 'independent'):
            model, memory, records, seen = make_model(p, seed, kind), empty_memory(p, seed, kind == 'shared'), [], []
            for position, value in enumerate(world.order[:4]):
                task = int(value)
                seen.append(task)
                records.append(check_stage(p, seed, world, study / f'seed_{seed}' / ('source_' + kind) / f'stage_{position}',
                                           model, memory, world.train[task], seen, p['source_steps'], kind == 'shared', counters))
            sources[kind] = model, memory, records
            source_rows.append({'seed': seed, 'kind': kind, 'records': records})
        for size in p['sample_sizes']:
            paired = {}
            for method in METHODS:
                whole = method in ('reuse', 'replay', 'independent')
                kind = 'independent' if method == 'independent' else 'shared'
                source_model, source_memory, source_records = sources[kind] if whole else (None, None, [])
                model = copy.deepcopy(source_model) if whole else make_model(p, seed)
                memory = copy.deepcopy(source_memory) if method == 'replay' else empty_memory(p, seed)
                if method in ('reuse', 'random_features'):
                    model.freeze_features()
                initial = model_digest(model)
                seen = list(map(int, world.order[:4])) if whole else []
                records, fresh_targets = [], {}
                folder = study / f'seed_{seed}' / f'samples_{size}' / method
                for position, value in enumerate(world.order[4:]):
                    task = int(value)
                    if method == 'fresh':
                        model, seen = make_model(p, seed), []
                    seen.append(task)
                    stage = folder / f'stage_{position}'
                    row = check_stage(p, seed, world, stage, model, memory, prefix(world.train[task], size),
                                      seen, p['target_steps'], method == 'replay', counters)
                    records.append(row)
                    fresh_targets[str(task)] = row['scores']['test'][str(task)]
                    if method != 'replay':
                        ids = np.load(stage / 'batch_ids.npy', allow_pickle=False)
                        if position in paired:
                            if not np.array_equal(ids, paired[position]):
                                raise AssertionError('Paired current-only target batches differ')
                            counters['paired_current_batches_checked'] += len(ids)
                        else:
                            paired[position] = ids
                targets = fresh_targets if method == 'fresh' else {str(int(t)): records[-1]['scores']['test'][str(int(t))] for t in world.order[4:]}
                sources_before = source_records[-1]['scores']['test'] if whole else None
                sources_after = {str(int(t)): records[-1]['scores']['test'][str(int(t))] for t in world.order[:4]} if whole else None
                unchanged = None
                if whole:
                    comparisons = []
                    with torch.no_grad():
                        for t in world.order[:4]:
                            data = world.test[int(t)]
                            comparisons.append(torch.equal(source_model(data.x, torch.from_numpy(data.task_ids)).argmax(1),
                                                           model(data.x, torch.from_numpy(data.task_ids)).argmax(1)))
                            counters['source_prediction_pairs_checked'] += 1
                    unchanged = all(comparisons)
                expected = {
                    'seed': seed, 'method': method, 'sample_size': size, 'whole_stream': whole,
                    'source_reference': kind if whole else None, 'initial_model_sha256': initial,
                    'target_test_by_task': targets, 'target_mean_accuracy': float(np.mean(list(targets.values()))),
                    'source_test_before': sources_before, 'source_test_final': sources_after,
                    'source_predictions_unchanged': unchanged,
                    'source_accuracy_drop': float(np.mean(list(sources_before.values())) - np.mean(list(sources_after.values()))) if whole else None,
                    'source_cost': cost(source_records), 'target_cost': cost(records),
                    'complete_cost': cost(source_records + records) if whole else None,
                    'parameter_stats': stats(p, kind, method in ('reuse', 'random_features')),
                    'source_training_labels_available': 4 * p['train_examples'] if whole else 0,
                    'target_training_labels_available': 2 * size,
                    'target_unique_labels_sampled': sum(row['unique_current_draws'] for row in records),
                }
                if read(folder / 'result.json') != expected:
                    raise AssertionError('Branch summary differs from checkpoint evidence')
                collected.append(expected)
            reference, weights = linear_reference(p, world, size)
            folder = study / f'seed_{seed}' / f'samples_{size}'
            saved_reference = read(folder / 'linear.json')
            with np.load(folder / 'linear_weights.npz', allow_pickle=False) as saved:
                if set(saved.files) != set(weights) or any(not np.allclose(saved[key], value, rtol=0, atol=1e-12) for key, value in weights.items()):
                    raise AssertionError('Linear reference does not reproduce from allowed labels')
            for key in ('sample_size', 'target_test_by_task', 'target_mean_accuracy', 'fitted_parameters'):
                if reference[key] != saved_reference[key]:
                    raise AssertionError('Linear reference scores differ')
            if saved_reference['seed'] != seed:
                raise AssertionError('Linear reference has the wrong seed')
            all_references.append(saved_reference)
            counters['linear_references_reproduced'] += 1
    key = lambda row: (row['seed'], row['sample_size'], row['method'])
    if sorted(collected, key=key) != sorted(summary['results'], key=key):
        raise AssertionError('Study outcomes differ from branch outcomes')
    if source_rows != summary['source_records'] or all_references != summary['linear_references']:
        raise AssertionError('Study references differ from their evidence')
    if summary['completed_branches'] != len(collected) or summary['source_models'] != len(source_rows):
        raise AssertionError('Study completion counts are wrong')
    check_decision_consistency(p, summary)
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    require_limits()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(json.dumps(verify_study(args.study), indent=2), flush=True)


if __name__ == '__main__':
    main()
