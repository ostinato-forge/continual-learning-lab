"""Freeze and run the declared shared-feature transfer study."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from continual_lab import digest, write_json
from transfer_lab import METHODS, decisions, fit_targets, linear_reference, make_world, train_source, validate_plan, world_arrays


ROOT = Path(__file__).resolve().parent


def observed_seeds(root=ROOT):
    values = set()
    for name in ('runs', 'studies'):
        for path in (root / name).rglob('config.json'):
            values.update(json.loads(path.read_text()).get('seeds', []))
        for path in (root / name).glob('*/plan.json'):
            values.update(json.loads(path.read_text()).get('seeds', []))
    return values


def require_limits():
    name = os.environ.get('CLL_RESOURCE_LIMITS')
    if not name or not Path(name).is_file():
        raise RuntimeError('Start the study through limited_run.py')
    limits = json.loads(Path(name).read_text())
    if (limits['cpu_percent_of_one_core'] > 50 or limits['library_threads'] != 1
            or limits['address_space_limit_mib_per_process'] > 2048
            or limits['resident_watchdog_mib_per_group'] > 1024
            or limits['minimum_available_memory_mib'] < 4096):
        raise RuntimeError('Keep the existing CPU and memory limits')
    return limits


def source_paths():
    return sorted(list(ROOT.glob('*.py')) + list(ROOT.glob('*.md'))
                  + list((ROOT / 'configs').glob('*.json')) + list((ROOT / 'tests').glob('*.py')))


def run_study(plan_path, output_root, pipeline=False):
    limits = require_limits()
    p = json.loads(plan_path.read_text())
    validate_plan(p)
    seen = observed_seeds()
    if set(p['seeds']) & seen:
        raise ValueError('A requested seed already has scored or incomplete study evidence')
    if pipeline and ROOT / 'checks' not in output_root.resolve().parents and output_root.resolve() != ROOT / 'checks':
        raise ValueError('Pipeline evidence belongs under checks')
    if not pipeline and output_root.resolve() != ROOT / 'studies':
        raise ValueError('Scored studies belong under studies')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    study = output_root / (stamp + '-transfer-' + uuid.uuid4().hex[:8])
    study.mkdir(parents=True, exist_ok=False)
    started, deadline = time.perf_counter(), time.perf_counter() + p['wall_seconds_limit']

    def emit(event, **details):
        with (study / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **details}) + '\n')
        print(event + ': ' + json.dumps(details), flush=True)

    completed = 0
    emit('study_started', path=str(study), purpose='pipeline' if pipeline else 'scored')
    try:
        write_json(study / 'plan.json', p)
        write_json(study / 'resource_limits.json', limits)
        write_json(study / 'seed_audit.json', {'previously_observed': sorted(seen), 'unobserved': True, 'purpose': 'pipeline' if pipeline else 'scored'})
        write_json(study / 'environment.json', {
            'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'platform': platform.platform(), 'executable': sys.executable, 'device': 'cpu', 'cpu_threads': 1,
        })
        frozen = {}
        for path in source_paths():
            target = study / 'source' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            frozen[str(target.relative_to(study))] = digest(target)
        worlds = {}
        (study / 'data').mkdir()
        for seed in p['seeds']:
            worlds[seed] = make_world(p, seed)
            path = study / 'data' / f'seed_{seed}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, **world_arrays(worlds[seed]))
            frozen[str(path.relative_to(study))] = digest(path)
        for name in ('plan.json', 'environment.json', 'resource_limits.json', 'seed_audit.json'):
            frozen[name] = digest(study / name)
        write_json(study / 'manifest.json', frozen)
        emit('inputs_frozen', manifest_sha256=digest(study / 'manifest.json'))
        results, references, source_costs = [], [], []
        for seed_index, seed in enumerate(p['seeds']):
            world = worlds[seed]
            sources = {}
            for kind in ('shared', 'independent'):
                emit('source_started', seed=seed, kind=kind)
                sources[kind] = train_source(p, seed, world, kind, study / f'seed_{seed}' / ('source_' + kind), deadline)
                source_costs.append({'seed': seed, 'kind': kind, 'records': sources[kind][2]})
                emit('source_completed', seed=seed, kind=kind)
            for size_index, size in enumerate(p['sample_sizes']):
                methods = list(METHODS)
                shift = (seed_index + size_index) % len(methods)
                methods = methods[shift:] + methods[:shift]
                for method in methods:
                    emit('targets_started', seed=seed, method=method, samples=size)
                    source = sources['independent' if method == 'independent' else 'shared']
                    output = study / f'seed_{seed}' / f'samples_{size}' / method
                    row = fit_targets(p, seed, world, method, size, source, output, deadline)
                    results.append(row)
                    completed += 1
                    emit('targets_completed', seed=seed, method=method, samples=size, target_accuracy=row['target_mean_accuracy'], completed_branches=completed)
                row, weights = linear_reference(p, world, size)
                row['seed'] = seed
                references.append(row)
                folder = study / f'seed_{seed}' / f'samples_{size}'
                write_json(folder / 'linear.json', row)
                with (folder / 'linear_weights.npz').open('xb') as stream:
                    np.savez(stream, **weights)
        summary = {
            'status': 'completed', 'purpose': 'pipeline' if pipeline else 'scored',
            'completed_branches': completed, 'source_models': 2 * len(p['seeds']),
            'results': results, 'linear_references': references, 'source_records': source_costs,
            'decision': decisions(p, results), 'wall_seconds': time.perf_counter() - started,
            'peak_process_memory_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'manifest_sha256': digest(study / 'manifest.json'),
        }
        write_json(study / 'summary.json', summary)
        emit('study_completed', completed_branches=completed, wall_seconds=summary['wall_seconds'],
             primary_successes=summary['decision']['passing_count'], criterion_passed=summary['decision']['passed'])
        write_json(study / 'artifacts.json', {str(path.relative_to(study)): digest(path) for path in study.rglob('*') if path.is_file()})
        return study
    except BaseException as exc:
        failure = {'status': 'incomplete' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else 'failed',
                   'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                   'completed_branches': completed, 'wall_seconds': time.perf_counter() - started}
        write_json(study / 'failure.json', failure)
        emit('study_stopped', **failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'studies')
    parser.add_argument('--pipeline', action='store_true')
    args = parser.parse_args()
    print(run_study(args.plan, args.output, args.pipeline), flush=True)


if __name__ == '__main__':
    main()
