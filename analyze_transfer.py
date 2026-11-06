"""Verify the transfer study and produce a report without changing evidence."""

import argparse
import csv
import json
from pathlib import Path
import shutil
from statistics import mean, stdev

import numpy as np
import torch

from continual_lab import digest, write_json
from transfer_lab import METHODS, decisions, stats
from transfer_study import require_limits
from verify_transfer import read, verify_study


LABELS = {'reuse': 'Learned frozen features', 'random_features': 'Random frozen features',
          'fresh': 'Fresh model', 'replay': 'Fixed 50% replay', 'independent': 'Independent networks',
          'linear': 'Linear reference'}


def spread(values, scale=100):
    return f'{scale * mean(values):.2f} ± {scale * (stdev(values) if len(values) > 1 else 0):.2f}'


def make_report(study, output=None):
    verified = verify_study(study)
    p, summary = read(study / 'plan.json'), read(study / 'summary.json')
    outcome = decisions(p, summary['results'])
    output = output or study / 'report'
    output.mkdir(parents=True, exist_ok=False)
    rows = summary['results']
    write_json(output / 'verification.json', verified)
    write_json(output / 'decisions.json', outcome)
    write_json(output / 'results.json', {'plan': p, 'results': rows, 'linear_references': summary['linear_references']})
    fields = ['seed', 'method', 'sample_size', 'target_mean_accuracy',
              'source_accuracy_before', 'source_accuracy_final', 'source_accuracy_drop',
              'source_training_labels_available', 'target_training_labels_available',
              'target_unique_labels_sampled', 'source_training_cpu_seconds', 'target_training_cpu_seconds',
              'complete_training_cpu_seconds', 'source_training_dense_macs', 'target_training_dense_macs',
              'complete_training_dense_macs', 'total_parameters', 'peak_optimizer_tensor_bytes',
              'peak_retained_examples']
    with (output / 'results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in fields if key in row}
            for phase in ('before', 'final'):
                scores = row['source_test_' + phase]
                record['source_accuracy_' + phase] = mean(scores.values()) if scores else None
            for phase in ('source', 'target', 'complete'):
                for metric in ('training_cpu_seconds', 'training_dense_macs'):
                    record[phase + '_' + metric] = row[phase + '_cost'][metric] if row[phase + '_cost'] else None
            record['total_parameters'] = row['parameter_stats']['total_parameters']
            total = row['complete_cost'] or row['target_cost']
            record['peak_optimizer_tensor_bytes'] = total['peak_optimizer_tensor_bytes']
            record['peak_retained_examples'] = total['peak_retained_examples']
            writer.writerow(record)

    candidate_n, reference_n = p['candidate_samples'], p['reference_samples']
    status = 'passed' if outcome['passed'] else 'failed'
    lines = ['# Shared feature transfer study', '', f'Study: {study.name}', '',
             f'Learned frozen features {status} the fixed transfer criterion: {outcome["passing_count"]} of {outcome["seed_count"]} seeds met every target; {outcome["required"]} were required.', '',
             f'The comparison uses {candidate_n} labelled training examples per target for learned features and {reference_n} for the fresh reference. Random features use {candidate_n}. Source learning uses {4 * p["train_examples"]} labelled examples per seed and is charged separately.', '',
             'The decision requires at least 80 percent test accuracy on each new task, at most a two-point mean deficit to the fresh reference, at least a five-point advantage over random features, and unchanged source predictions. These numeric thresholds are the scored protocol; pipeline settings are shown in plan.json.', '',
             'All sample sizes and comparisons were fixed before scoring. No result selected another seed, threshold, or sample size. Values below are means and sample standard deviations across seeds.', '',
             '| Method | Target examples per task | Mean target test accuracy |', '| --- | ---: | ---: |']
    if summary['purpose'] == 'pipeline':
        lines[2:2] = ['This is a tiny pipeline check. Its scores are not research evidence.', '']
    for n in p['sample_sizes']:
        for method in METHODS:
            values = [r['target_mean_accuracy'] for r in rows if r['sample_size'] == n and r['method'] == method]
            lines.append(f'| {LABELS[method]} | {n} | {spread(values)}% |')
        values = [r['target_mean_accuracy'] for r in summary['linear_references'] if r['sample_size'] == n]
        lines.append(f'| Linear reference | {n} | {spread(values)}% |')
    lines += ['', 'The linear reference fits separate raw-input classifiers for the two targets. It does not learn the four source tasks. Its measured time below includes both fitting and test prediction, so it differs from the neural training-only timing.', '',
              '| Target examples per task | Fitted parameters across two targets | Mean fit and prediction CPU seconds |',
              '| ---: | ---: | ---: |']
    for n in p['sample_sizes']:
        selected = [r for r in summary['linear_references'] if r['sample_size'] == n]
        lines.append(f'| {n} | {selected[0]["fitted_parameters"]} | {mean(r["cpu_seconds"] for r in selected):.6f} |')
    lines += ['', '## Primary decisions by seed', '',
              '| Seed | Learned features | Fresh reference | Random features | Failed conditions |',
              '| --- | ---: | ---: | ---: | --- |']
    for row in outcome['pairs']:
        failed = ', '.join(row['failed_targets']) if row['failed_targets'] else 'none'
        lines.append(f'| {row["seed"]} | {100 * row["candidate_mean_accuracy"]:.4f}% | {100 * row["fresh_reference_accuracy"]:.4f}% | {100 * row["random_control_accuracy"]:.4f}% | {failed} |')
    lines += ['', 'decisions.json retains the exact values and each target score. Rounding never changes a decision. The seed count is a working criterion, not a significance test.', '',
              '## Source retention and complete learning cost', '',
              f'This table uses {candidate_n} target examples per task in all three complete-stream methods. It charges the source phase once to each method. Fresh and random-feature controls did not learn and retain the source tasks, so they are excluded from this full-stream comparison.', '',
              '| Method | Source accuracy before targets | Source accuracy after targets | Earlier accuracy lost in points |',
              '| --- | ---: | ---: | ---: |']
    for method in ('reuse', 'replay', 'independent'):
        selected = [r for r in rows if r['method'] == method and r['sample_size'] == candidate_n]
        before = spread([mean(r['source_test_before'].values()) for r in selected])
        final = spread([mean(r['source_test_final'].values()) for r in selected])
        lines.append(f'| {LABELS[method]} | {before}% | {final}% | {spread([r["source_accuracy_drop"] for r in selected])} |')
    lines += ['', '| Method | Source training CPU seconds | Target training CPU seconds | Total training dense MACs |',
              '| --- | ---: | ---: | ---: |']
    for method in ('reuse', 'replay', 'independent'):
        selected = [r for r in rows if r['method'] == method and r['sample_size'] == candidate_n]
        lines.append(f'| {LABELS[method]} | {mean(r["source_cost"]["training_cpu_seconds"] for r in selected):.3f} | {mean(r["target_cost"]["training_cpu_seconds"] for r in selected):.3f} | {mean(r["complete_cost"]["training_dense_macs"] for r in selected):,.0f} |')
    lines += ['', 'MACs are dense-layer multiply-add operations in forward and backward matrix products. They exclude biases, activations, routing, loss, optimizer updates, and memory work. CPU timing includes the measured optimizer work but remains subject to machine conditions. Neither is an energy measurement.', '',
              'The two source models are physically trained once per seed and copied into independent sample-size branches. Costs are attributed once per complete scenario, rather than treating copied source learning as free. Adding all scenario costs would double count shared execution. Overall study time is reported separately.', '',
              '## Paired comparisons at the fixed sample size', '',
              f'Learned frozen features at {candidate_n} examples are compared below with references at {reference_n}. Differences are learned features minus the reference.', '',
              '| Reference | Target accuracy difference in points | Complete training dense MAC difference |', '| --- | ---: | ---: |']
    index = {(r['seed'], r['method'], r['sample_size']): r for r in rows}
    for method in ('fresh', 'replay', 'independent'):
        deltas, arithmetic = [], []
        for seed in p['seeds']:
            a, b = index[seed, 'reuse', candidate_n], index[seed, method, reference_n]
            deltas.append(a['target_mean_accuracy'] - b['target_mean_accuracy'])
            if b['whole_stream']:
                arithmetic.append(a['complete_cost']['training_dense_macs'] - b['complete_cost']['training_dense_macs'])
        mac_text = f'{mean(arithmetic):+,.0f}' if arithmetic else 'different source-task scope'
        lines.append(f'| {LABELS[method]} | {spread(deltas)} | {mac_text} |')
    lines += ['', 'A lower arithmetic count is not a useful-cost win when the method delivers worse accuracy or retention. The primary criterion tests later labelled-data use. It does not prove lower total learning cost.', '',
              '## Resources and evidence', '',
              f'The study completed {summary["source_models"]} source models and {summary["completed_branches"]} target branches in {summary["wall_seconds"]:.1f} seconds. Each branch contains two target fits. Peak process memory was {summary["peak_process_memory_mib"]:.1f} MiB.', '',
              f'Verification regenerated {verified["datasets_regenerated"]} datasets, reproduced {verified["checkpoints_reproduced"]} task-end checkpoints, and audited {verified["training_example_draws_checked"]:,} training-example draws. It checked {verified["protected_components_checked"]:,} protected components, {verified["paired_current_batches_checked"]:,} matching current-only batches, and {verified["source_prediction_pairs_checked"]:,} source prediction pairs. It reproduced {verified["linear_references_reproduced"]} linear references from the allowed labels.', '',
              'Verification reads saved model states and reproduces their predictions. It audits deterministic draw traces and parameter protection. It does not retrain every optimizer update.', '',
              '| Architecture | Total parameters | Active parameters | Parameter bytes |', '| --- | ---: | ---: | ---: |']
    for kind in ('shared', 'independent'):
        item = stats(p, kind)
        lines.append(f'| {kind} | {item["total_parameters"]:,} | {item["active_parameters"]:,} | {item["parameter_bytes"]:,} |')
    lines += ['', 'results.csv records source and target cost, available and sampled labels, optimizer tensor bytes, and replay counts. The branch stage records also separate sampling, optimizer setup, memory work, evaluation, and checkpoint capture. These are measured sections, not all process work. They exclude model construction, branch copies, final prediction comparisons, and some audit and file-writing work. Overall study time includes that work. Learner-state counts exclude the experiment harness datasets and saved evidence; process memory includes them.', '',
              'The launcher used one library thread, half of one CPU at low priority, a 2 GiB address-space limit, a 1 GiB resident-memory watchdog, and a 4 GiB available-memory reserve. Its own receipt records sampled CPU and memory and any stop reason.', '',
              '## Interpretation limits', '',
              'The tasks deliberately share four hidden directions. The target pairs are new combinations of those directions. Classifiers still receive supervised labels and supplied task identities. The feature extractor stops learning after four source tasks, so this tests bounded transfer, not continued feature learning over an unlimited stream.', '',
              'All new methods use the same optimizer reset rule at task boundaries. The fresh reference resets its model for each target and does not retain old tasks. The linear reference has a different fitting procedure and must remain visible when judging whether the neural system is worthwhile.', '',
              'No outcome establishes language learning, a novel continual-learning method, or a replacement for large-model pre-training. Earlier studies and their negative outcomes remain unchanged.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {'reuse': '#176b49', 'random_features': '#7d7d7d', 'fresh': '#4477aa',
              'replay': '#cc6677', 'independent': '#aa7733', 'linear': '#8855aa'}
    for method in (*METHODS, 'linear'):
        groups = [[r['target_mean_accuracy'] for r in (summary['linear_references'] if method == 'linear' else rows)
                   if r['sample_size'] == n and (method == 'linear' or r['method'] == method)] for n in p['sample_sizes']]
        axes[0].errorbar(p['sample_sizes'], [100 * mean(g) for g in groups],
                         yerr=[100 * stdev(g) if len(g) > 1 else 0 for g in groups],
                         marker='o', color=colors[method], label=LABELS[method], capsize=3)
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(p['sample_sizes'], [str(n) for n in p['sample_sizes']])
    axes[0].set(xlabel='Labelled target examples per task', ylabel='Mean target test accuracy (%)',
                ylim=(0, 100), title='Fixed sample sizes, all results shown')
    axes[0].axhline(p['target_accuracy'] * 100, color='#555555', linestyle=':', linewidth=1)
    axes[0].legend(fontsize=8, loc='lower right')
    selected_methods = ('reuse', 'replay', 'independent')
    source_values, target_values = [], []
    for method in selected_methods:
        selected = [r for r in rows if r['method'] == method and r['sample_size'] == candidate_n]
        source_values.append(mean(r['source_cost']['training_cpu_seconds'] for r in selected))
        target_values.append(mean(r['target_cost']['training_cpu_seconds'] for r in selected))
    axes[1].bar(range(3), source_values, color='#a7b9c8', label='Source learning')
    axes[1].bar(range(3), target_values, bottom=source_values, color='#176b49', label='Target learning')
    axes[1].set_xticks(range(3), ['Frozen reuse', 'Fixed replay', 'Independent'])
    axes[1].set(ylabel='Mean training CPU seconds', title=f'Initial learning included, {candidate_n} target examples')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('Shared feature transfer' + (' (pipeline check)' if summary['purpose'] == 'pipeline' else ''))
    fig.tight_layout()
    fig.savefig(output / 'results.png', dpi=160)
    fig.savefig(output / 'results.svg')
    plt.close(fig)
    shutil.copy2(Path(__file__), output / 'analysis_source.py')
    write_json(output / 'provenance.json', {
        'study_artifacts_sha256': digest(study / 'artifacts.json'),
        'analysis_source_sha256': digest(output / 'analysis_source.py'),
        'files_sha256': {path.name: digest(path) for path in output.iterdir() if path.is_file()},
    })
    print(json.dumps({'report': str(output), 'verification': verified, 'decision': outcome}, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require_limits()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    make_report(args.study, args.output)


if __name__ == '__main__':
    main()
