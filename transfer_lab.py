"""Fixed shared-rule data and bounded feature-transfer comparisons."""

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from continual_lab import Examples, ReplayMemory, model_digest, write_json


METHODS = ('reuse', 'random_features', 'fresh', 'replay', 'independent')
PLAN_KEYS = {
    'experiment', 'seeds', 'features', 'classes', 'source_pairs', 'target_pairs',
    'train_examples', 'validation_examples', 'test_examples', 'sample_sizes',
    'shared_widths', 'expert_widths', 'parameter_budget', 'max_parameter_difference',
    'source_steps', 'target_steps', 'batch_size', 'learning_rate', 'weight_decay',
    'memory_capacity', 'replay_fraction', 'ridge_penalty', 'candidate_samples',
    'reference_samples', 'target_accuracy', 'accuracy_tolerance',
    'random_feature_margin', 'required_successes', 'cpu_threads', 'wall_seconds_limit',
}


def stats(p, kind='shared', frozen=False):
    a, b = p['expert_widths'] if kind == 'independent' else p['shared_widths']
    f, k = p['features'], p['classes']
    tasks = len(p['source_pairs']) + len(p['target_pairs'])
    trunk = (f + 1) * a + (a + 1) * b
    head = (b + 1) * k
    total = tasks * (trunk + head) if kind == 'independent' else trunk + tasks * head
    forward = f * a + a * b + b * k
    train = forward + b * k if frozen else 2 * f * a + 3 * a * b + 3 * b * k
    return {
        'total_parameters': total, 'active_parameters': trunk + head,
        'parameter_bytes': total * 4, 'forward_dense_macs_per_example': forward,
        'training_dense_macs_per_example': train,
    }


def validate_plan(p):
    if set(p) != PLAN_KEYS:
        raise ValueError('Unexpected transfer plan keys')
    for key in ('features', 'classes', 'train_examples', 'validation_examples',
                'test_examples', 'parameter_budget', 'source_steps', 'target_steps',
                'batch_size', 'memory_capacity', 'candidate_samples',
                'reference_samples', 'required_successes', 'wall_seconds_limit'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError('Invalid positive integer: ' + key)
    if p['features'] < 4 or p['classes'] != 4 or p['cpu_threads'] != 1:
        raise ValueError('Use four latent directions, four classes, and one CPU thread')
    if p['source_pairs'] != [[0, 1], [1, 2], [2, 3], [3, 0]] or p['target_pairs'] != [[0, 2], [1, 3]]:
        raise ValueError('The shared-rule pairs are fixed')
    seeds = p['seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    if set(seeds) & {401, 433, 463, 499, 541}:
        raise ValueError('Earlier confirmation seeds remain reserved')
    sizes = p['sample_sizes']
    if not sizes or sorted(set(sizes)) != sizes or any(type(n) is not int or not 1 <= n <= p['train_examples'] for n in sizes):
        raise ValueError('Sample sizes must be sorted unique training prefixes')
    if not p['candidate_samples'] < p['reference_samples'] or not {p['candidate_samples'], p['reference_samples']} <= set(sizes):
        raise ValueError('Decision sample sizes must be present and ordered')
    if not 1 <= p['required_successes'] <= len(seeds):
        raise ValueError('Success count must fit the seed count')
    for key in ('shared_widths', 'expert_widths'):
        if len(p[key]) != 2 or any(type(v) is not int or v < 1 for v in p[key]):
            raise ValueError('Use two positive hidden widths')
    if type(p['max_parameter_difference']) is not int or p['max_parameter_difference'] < 0:
        raise ValueError('Invalid parameter difference')
    counts = [stats(p, kind)['total_parameters'] for kind in ('shared', 'independent')]
    if max(counts) > p['parameter_budget'] or abs(counts[0] - counts[1]) > p['max_parameter_difference']:
        raise ValueError('Architecture exceeds the fixed parameter comparison')
    if p['learning_rate'] <= 0 or p['weight_decay'] < 0 or p['ridge_penalty'] <= 0:
        raise ValueError('Invalid optimizer or reference setting')
    if not 0 < p['replay_fraction'] < 1 or not float(p['batch_size'] * p['replay_fraction']).is_integer():
        raise ValueError('Replay must use a whole positive part of each batch')
    if not 0 < p['target_accuracy'] <= 1:
        raise ValueError('Invalid target accuracy')
    if any(not 0 <= p[key] <= 1 for key in ('accuracy_tolerance', 'random_feature_margin')):
        raise ValueError('Invalid accuracy comparison margin')


@dataclass
class TransferWorld:
    train: list
    validation: list
    test: list
    directions: np.ndarray
    label_maps: np.ndarray
    order: np.ndarray


def make_world(p, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8100]))
    q, _ = np.linalg.qr(rng.normal(size=(p['features'], 4)))
    directions = q.T.astype(np.float32)
    pairs = p['source_pairs'] + p['target_pairs']
    maps = np.stack([rng.permutation(4) for _ in pairs])
    order_rng = np.random.default_rng(np.random.SeedSequence([seed, 8101]))
    order = np.concatenate([order_rng.permutation(4), 4 + order_rng.permutation(2)])
    groups = {'train': [], 'validation': [], 'test': []}
    offset, rows = 0, set()
    for split_id, (split, values) in enumerate(groups.items()):
        for task, pair in enumerate(pairs):
            local = np.random.default_rng(np.random.SeedSequence([seed, 8200, split_id, task]))
            x = local.uniform(-1, 1, (p[split + '_examples'], p['features'])).astype(np.float32)
            for row in x:
                if row.tobytes() in rows:
                    raise AssertionError('Duplicate raw input across splits')
                rows.add(row.tobytes())
            projected = x @ directions[pair].T
            bins = (projected[:, 0] > 0).astype(np.int64) + 2 * (projected[:, 1] > 0)
            y = maps[task][bins]
            values.append(Examples(torch.from_numpy(x), torch.from_numpy(y.copy()),
                                   np.arange(offset, offset + len(x), dtype=np.int64),
                                   np.full(len(x), task, dtype=np.int64), split))
            offset += len(x)
    return TransferWorld(**groups, directions=directions, label_maps=maps, order=order)


def world_arrays(world):
    arrays = {'directions': world.directions, 'label_maps': world.label_maps, 'order': world.order}
    for split in ('train', 'validation', 'test'):
        for task, data in enumerate(getattr(world, split)):
            for name, value in (('x', data.x.numpy()), ('y', data.y.numpy()), ('ids', data.ids)):
                arrays[f'{split}_{task}_{name}'] = value
    return arrays


def prefix(data, count):
    if data.split != 'train' or not 1 <= count <= len(data.ids):
        raise ValueError('Only a valid training prefix is allowed')
    return Examples(data.x[:count], data.y[:count], data.ids[:count], data.task_ids[:count], 'train')


class TransferNetwork(nn.Module):
    def __init__(self, p, kind='shared'):
        super().__init__()
        self.kind = kind
        tasks = len(p['source_pairs']) + len(p['target_pairs'])
        a, b = p['expert_widths'] if kind == 'independent' else p['shared_widths']
        if kind == 'independent':
            self.parts = nn.ModuleList([
                nn.Sequential(nn.Linear(p['features'], a), nn.ReLU(), nn.Linear(a, b), nn.ReLU(), nn.Linear(b, p['classes']))
                for _ in range(tasks)
            ])
        else:
            self.trunk = nn.Sequential(nn.Linear(p['features'], a), nn.ReLU(), nn.Linear(a, b), nn.ReLU())
            self.parts = nn.ModuleList([nn.Linear(b, p['classes']) for _ in range(tasks)])

    def forward(self, x, task_ids):
        if len(x) != len(task_ids) or bool(torch.any((task_ids < 0) | (task_ids >= len(self.parts)))):
            raise ValueError('Every input needs a valid supplied task identity')
        features = x if self.kind == 'independent' else self.trunk(x)
        output = x.new_zeros((len(x), 4))
        for task in torch.unique(task_ids).tolist():
            indexes = torch.nonzero(task_ids == task, as_tuple=False).flatten()
            output = output.index_copy(0, indexes, self.parts[task](features.index_select(0, indexes)))
        return output

    def freeze_features(self):
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)

    def hashes(self):
        values = {f'task_{i}': model_digest(part) for i, part in enumerate(self.parts)}
        if self.kind != 'independent':
            values['trunk'] = model_digest(self.trunk)
        return values


def make_model(p, seed, kind='shared'):
    torch.manual_seed(seed)
    model = TransferNetwork(p, kind)
    if sum(v.numel() for v in model.parameters()) != stats(p, kind)['total_parameters']:
        raise AssertionError('Parameter formula does not match the model')
    return model


@torch.no_grad()
def evaluate(model, world, tasks):
    model.eval()
    result = {}
    for split in ('validation', 'test'):
        scores = {}
        for task in tasks:
            data = getattr(world, split)[task]
            prediction = model(data.x, torch.from_numpy(data.task_ids)).argmax(1)
            scores[str(task)] = float((prediction == data.y).float().mean().item())
        result[split] = scores
    return result


def batch_sequence(p, seed, task, current, old, steps, replay):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8300, task]))
    old_rng = np.random.default_rng(np.random.SeedSequence([seed, 8301, task]))
    count_old = int(p['batch_size'] * p['replay_fraction']) if replay and len(old.ids) else 0
    count_new = p['batch_size'] - count_old
    for _ in range(steps):
        indexes = rng.integers(len(current.ids), size=p['batch_size'])[:count_new]
        old_indexes = old_rng.integers(len(old.ids), size=count_old) if count_old else np.empty(0, dtype=np.int64)
        yield (
            torch.cat([current.x[indexes], old.x[old_indexes]]),
            torch.cat([current.y[indexes], old.y[old_indexes]]),
            np.concatenate([current.ids[indexes], old.ids[old_indexes]]),
            np.concatenate([current.task_ids[indexes], old.task_ids[old_indexes]]),
        )


def empty_memory(p, seed, enabled=False):
    return ReplayMemory(p['memory_capacity'] if enabled else 0, p['features'], seed + 8400)


def fit_stage(p, seed, model, current, memory, task, steps, replay, world, seen, output, deadline):
    output.mkdir(parents=True, exist_ok=False)
    before = model.hashes()
    old = memory.examples()
    old_ids = old.ids.copy()
    allowed = set(current.ids.tolist()) | (set(old_ids.tolist()) if replay else set())
    wall, cpu = time.perf_counter(), time.process_time()
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],
                                 lr=p['learning_rate'], weight_decay=p['weight_decay'])
    setup_wall, setup_cpu = time.perf_counter() - wall, time.process_time() - cpu
    batches, unique = [], set()
    train_wall = train_cpu = 0.0
    sampling_wall = sampling_cpu = 0.0
    iterator = iter(batch_sequence(p, seed, task, current, old, steps, replay))
    for _ in range(steps):
        wall, cpu = time.perf_counter(), time.process_time()
        x, y, ids, task_ids = next(iterator)
        sampling_wall += time.perf_counter() - wall
        sampling_cpu += time.process_time() - cpu
        if time.perf_counter() > deadline:
            raise TimeoutError('The fixed transfer wall limit was reached')
        if not set(ids.tolist()) <= allowed or not set(task_ids.tolist()) <= set(seen):
            raise AssertionError('Training accessed unavailable examples')
        batches.append(ids)
        unique.update(ids[task_ids == task].tolist())
        wall, cpu = time.perf_counter(), time.process_time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x, torch.from_numpy(task_ids)), y)
        loss.backward()
        optimizer.step()
        train_wall += time.perf_counter() - wall
        train_cpu += time.process_time() - cpu
    state_bytes = sum(v.numel() * v.element_size() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
    after = model.hashes()
    frozen = model.kind == 'shared' and not next(model.trunk.parameters()).requires_grad
    for name in before:
        if (name == 'trunk' and frozen) or (name.startswith('task_') and int(name[5:]) not in (set(seen) if replay else {task})):
            if before[name] != after[name]:
                raise AssertionError('A protected component changed')
    wall, cpu = time.perf_counter(), time.process_time()
    if replay:
        memory.add(current)
    memory_wall, memory_cpu = time.perf_counter() - wall, time.process_time() - cpu
    wall, cpu = time.perf_counter(), time.process_time()
    scores = evaluate(model, world, seen)
    evaluation_wall, evaluation_cpu = time.perf_counter() - wall, time.process_time() - cpu
    count_old = int(p['batch_size'] * p['replay_fraction']) if replay and len(old_ids) else 0
    arithmetic = stats(p, model.kind, frozen)
    record = {
        'task': task, 'seen_tasks': list(seen), 'steps': steps, 'replay': replay,
        'frozen_features': frozen, 'initial_components': before, 'final_components': after,
        'model_sha256': model_digest(model), 'scores': scores,
        'current_available': len(current.ids), 'unique_current_draws': len(unique),
        'presentations': steps * p['batch_size'],
        'current_presentations': steps * (p['batch_size'] - count_old),
        'replay_presentations': steps * count_old,
        'training_dense_macs': steps * p['batch_size'] * arithmetic['training_dense_macs_per_example'],
        'evaluation_dense_macs': len(seen) * (p['validation_examples'] + p['test_examples']) * arithmetic['forward_dense_macs_per_example'],
        'training_wall_seconds': train_wall, 'training_cpu_seconds': train_cpu,
        'sampling_wall_seconds': sampling_wall, 'sampling_cpu_seconds': sampling_cpu,
        'optimizer_setup_wall_seconds': setup_wall, 'optimizer_setup_cpu_seconds': setup_cpu,
        'memory_wall_seconds': memory_wall, 'memory_cpu_seconds': memory_cpu,
        'evaluation_wall_seconds': evaluation_wall, 'evaluation_cpu_seconds': evaluation_cpu,
        'optimizer_tensor_bytes': state_bytes,
        'retained_examples': memory.size if replay else 0,
        'retained_example_bytes': (p['features'] * 4 + 8 + 8 + 8) * memory.size if replay else 0,
    }
    wall, cpu = time.perf_counter(), time.process_time()
    torch.save(model.state_dict(), output / 'model.pt')
    for name, value in (('batch_ids', np.stack(batches)), ('current_ids', current.ids),
                        ('memory_before_ids', old_ids), ('memory_after_ids', memory.examples().ids.copy())):
        with (output / (name + '.npy')).open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
    record['evidence_wall_seconds'] = time.perf_counter() - wall
    record['evidence_cpu_seconds'] = time.process_time() - cpu
    write_json(output / 'stage.json', record)
    return record


def cost(records):
    total = {}
    for key in ('steps', 'presentations', 'current_presentations', 'replay_presentations',
                'training_dense_macs', 'evaluation_dense_macs', 'training_wall_seconds',
                'training_cpu_seconds', 'memory_wall_seconds', 'memory_cpu_seconds',
                'sampling_wall_seconds', 'sampling_cpu_seconds', 'optimizer_setup_wall_seconds', 'optimizer_setup_cpu_seconds',
                'evaluation_wall_seconds', 'evaluation_cpu_seconds', 'evidence_wall_seconds', 'evidence_cpu_seconds'):
        total[key] = sum(row[key] for row in records)
    total['peak_optimizer_tensor_bytes'] = max((row['optimizer_tensor_bytes'] for row in records), default=0)
    total['peak_retained_examples'] = max((row['retained_examples'] for row in records), default=0)
    total['peak_retained_example_bytes'] = max((row['retained_example_bytes'] for row in records), default=0)
    return total


def train_source(p, seed, world, kind, output, deadline):
    model, memory = make_model(p, seed, kind), empty_memory(p, seed, kind == 'shared')
    records, seen = [], []
    for position, task_value in enumerate(world.order[:4]):
        task = int(task_value)
        seen.append(task)
        records.append(fit_stage(p, seed, model, world.train[task], memory, task,
                                 p['source_steps'], kind == 'shared', world, seen,
                                 output / f'stage_{position}', deadline))
    return model, memory, records


def fit_targets(p, seed, world, method, sample_size, source, output, deadline):
    kind = 'independent' if method == 'independent' else 'shared'
    whole_stream = method in ('reuse', 'replay', 'independent')
    source_model, source_memory, source_records = source if whole_stream else (None, None, [])
    model = copy.deepcopy(source_model) if whole_stream else make_model(p, seed)
    memory = copy.deepcopy(source_memory) if method == 'replay' else empty_memory(p, seed)
    if method in ('reuse', 'random_features'):
        model.freeze_features()
    initial_hash = model_digest(model)
    source_scores = source_records[-1]['scores']['test'] if source_records else None
    records, seen, reference_targets = [], list(map(int, world.order[:4])) if whole_stream else [], {}
    for position, task_value in enumerate(world.order[4:]):
        task = int(task_value)
        if method == 'fresh':
            model, seen = make_model(p, seed), []
        seen.append(task)
        record = fit_stage(p, seed, model, prefix(world.train[task], sample_size), memory,
                           task, p['target_steps'], method == 'replay', world, seen,
                           output / f'stage_{position}', deadline)
        records.append(record)
        reference_targets[str(task)] = record['scores']['test'][str(task)]
    targets = reference_targets if method == 'fresh' else {str(int(t)): records[-1]['scores']['test'][str(int(t))] for t in world.order[4:]}
    final_sources = {str(int(t)): records[-1]['scores']['test'][str(int(t))] for t in world.order[:4]} if whole_stream else None
    unchanged = None
    if whole_stream:
        with torch.no_grad():
            unchanged = all(torch.equal(
                source_model(world.test[int(t)].x, torch.from_numpy(world.test[int(t)].task_ids)).argmax(1),
                model(world.test[int(t)].x, torch.from_numpy(world.test[int(t)].task_ids)).argmax(1),
            ) for t in world.order[:4])
    result = {
        'seed': seed, 'method': method, 'sample_size': sample_size, 'whole_stream': whole_stream,
        'source_reference': kind if whole_stream else None,
        'initial_model_sha256': initial_hash,
        'target_test_by_task': targets, 'target_mean_accuracy': float(np.mean(list(targets.values()))),
        'source_test_before': source_scores, 'source_test_final': final_sources,
        'source_predictions_unchanged': unchanged,
        'source_accuracy_drop': float(np.mean(list(source_scores.values())) - np.mean(list(final_sources.values()))) if whole_stream else None,
        'source_cost': cost(source_records), 'target_cost': cost(records),
        'complete_cost': cost(source_records + records) if whole_stream else None,
        'parameter_stats': stats(p, kind, method in ('reuse', 'random_features')),
        'source_training_labels_available': 4 * p['train_examples'] if whole_stream else 0,
        'target_training_labels_available': 2 * sample_size,
        'target_unique_labels_sampled': sum(row['unique_current_draws'] for row in records),
    }
    write_json(output / 'result.json', result)
    return result


def linear_reference(p, world, sample_size):
    scores, weights = {}, {}
    wall, cpu = time.perf_counter(), time.process_time()
    for task_value in world.order[4:]:
        task = int(task_value)
        data = prefix(world.train[task], sample_size)
        x = np.column_stack([data.x.numpy().astype(np.float64), np.ones(len(data.ids))])
        y = np.eye(4)[data.y.numpy()]
        penalty = p['ridge_penalty'] * np.eye(x.shape[1])
        w = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        test = np.column_stack([world.test[task].x.numpy(), np.ones(len(world.test[task].ids))])
        scores[str(task)] = float(np.mean((test @ w).argmax(1) == world.test[task].y.numpy()))
        weights[str(task)] = w
    return {'sample_size': sample_size, 'target_test_by_task': scores,
            'target_mean_accuracy': float(np.mean(list(scores.values()))),
            'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
            'fitted_parameters': 2 * (p['features'] + 1) * 4}, weights


def decisions(p, results):
    indexed = {(row['seed'], row['method'], row['sample_size']): row for row in results}
    expected = {(s, m, n) for s in p['seeds'] for m in METHODS for n in p['sample_sizes']}
    if len(indexed) != len(results) or set(indexed) != expected:
        raise AssertionError('Missing or duplicate transfer outcomes')
    pairs = []
    for seed in p['seeds']:
        reuse = indexed[seed, 'reuse', p['candidate_samples']]
        fresh = indexed[seed, 'fresh', p['reference_samples']]
        random = indexed[seed, 'random_features', p['candidate_samples']]
        targets = {
            'each_target_accuracy': all(v >= p['target_accuracy'] for v in reuse['target_test_by_task'].values()),
            'fresh_accuracy_tolerance': reuse['target_mean_accuracy'] >= fresh['target_mean_accuracy'] - p['accuracy_tolerance'],
            'learned_feature_advantage': reuse['target_mean_accuracy'] - random['target_mean_accuracy'] >= p['random_feature_margin'],
            'source_predictions_unchanged': reuse['source_predictions_unchanged'] is True,
        }
        pairs.append({'seed': seed, 'candidate_mean_accuracy': reuse['target_mean_accuracy'],
                      'candidate_target_accuracy': reuse['target_test_by_task'],
                      'fresh_reference_accuracy': fresh['target_mean_accuracy'],
                      'random_control_accuracy': random['target_mean_accuracy'],
                      'targets': targets, 'passes': all(targets.values()),
                      'failed_targets': [k for k, v in targets.items() if not v]})
    successes = sum(row['passes'] for row in pairs)
    return {'passed': successes >= p['required_successes'], 'passing_count': successes,
            'required': p['required_successes'], 'seed_count': len(p['seeds']), 'pairs': pairs,
            'target_counts': {key: sum(row['targets'][key] for row in pairs) for key in pairs[0]['targets']}}
