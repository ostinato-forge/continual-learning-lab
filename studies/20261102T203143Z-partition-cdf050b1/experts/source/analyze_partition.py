"""Verify the fixed partition study and report every scored condition."""

import argparse
import csv
import json
from pathlib import Path
import shutil
from statistics import mean

import numpy as np
import torch

from analyze_memory_sweep import dataset_fingerprint, fmt
from analyze_replay_allocation import audit_allocation
from continual_lab import digest, make_model, model_digest, write_json
from partition_models import architecture_stats
from partition_study import architecture_for, condition_config, validate_plan
from verify_partition import verify_partition
from verify_run import verify


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = (
    "continual_lab.py", "run_experiment.py", "verify_run.py", "partition_models.py",
    "partition_training.py", "partition_study.py", "verify_partition.py",
    "analyze_memory_sweep.py", "analyze_replay_allocation.py", "memory_sweep.py",
    "replay_allocation.py",
)
METRICS = ("final_mean_accuracy", "mean_acquisition_accuracy", "old_task_retention_drop")
LABELS = {
    "online": "Shared, current only", "replay": "Fixed 50% replay",
    "heads": "Separate output layers", "experts": "Independent networks",
    "full_history": "Full history",
}


def read(path):
    return json.loads(path.read_text())


def decisions(results, plan):
    expected = {(s, m) for s in plan["seeds"] for m in LABELS}
    indexed = {(r["seed"], r["method"]): r for r in results}
    if set(indexed) != expected or len(indexed) != len(results):
        raise AssertionError("Missing, duplicate, or unexpected scored conditions")
    outcomes = {}
    for method in ("replay", "heads", "experts"):
        pairs = []
        for seed in plan["seeds"]:
            row, baseline = indexed[seed, method], indexed[seed, "online"]
            final = row[METRICS[0]]
            retention = baseline[METRICS[2]] - row[METRICS[2]]
            acquisition = row[METRICS[1]] - baseline[METRICS[1]]
            targets = {
                "final_accuracy": final >= plan["final_accuracy_target"],
                "retention": retention >= plan["retention_margin"],
                "acquisition": acquisition >= -plan["acquisition_tolerance"],
            }
            pairs.append({
                "seed": seed, "final_mean_accuracy": final,
                "retention_improvement": retention, "acquisition_change": acquisition,
                "targets": targets, "passes": all(targets.values()),
                "failed_targets": [name for name, passed in targets.items() if not passed],
            })
        count = sum(p["passes"] for p in pairs)
        outcomes[method] = {
            "passed": count >= plan["required_successes"], "passing_count": count,
            "required": plan["required_successes"], "pairs": pairs,
            "target_counts": {name: sum(p["targets"][name] for p in pairs) for name in pairs[0]["targets"]},
        }
    for method in ("heads", "experts"):
        outcomes[method + "_minus_replay"] = [
            {"seed": seed, **{key: indexed[seed, method][key] - indexed[seed, "replay"][key] for key in METRICS}}
            for seed in plan["seeds"]
        ]
    return outcomes


def verify_study(study):
    plan, base = read(study / "plan.json"), read(study / "base_config.json")
    validate_plan(plan, base)
    manifest, summary = read(study / "manifest.json"), read(study / "study_summary.json")
    if summary["status"] != "completed" or digest(study / "manifest.json") != summary["manifest_sha256"]:
        raise AssertionError("The study is incomplete or its manifest changed")
    for name, key in (("plan.json", "plan_sha256"), ("base_config.json", "base_sha256"),
                      ("seed_audit.json", "seed_audit_sha256"), ("resource_limits.json", "resource_limits_sha256")):
        if digest(study / name) != manifest[key]:
            raise AssertionError(f"A frozen input changed: {name}")
    seed_audit = read(study / "seed_audit.json")
    if not seed_audit["study_seeds_were_unobserved"] or set(seed_audit["previously_observed"]) & set(plan["seeds"]):
        raise AssertionError("Scored seeds were previously observed")
    for name, checksum in manifest["source_sha256"].items():
        if digest(study / "source" / name) != checksum:
            raise AssertionError("Frozen source changed")
    for name in DEPENDENCIES:
        if digest(ROOT / name) != manifest["source_sha256"][name]:
            raise AssertionError(f"Use the frozen experiment code: {name}")
    checks, results, stages, fingerprints = [], [], [], {}
    references = None
    batch_comparisons = 0
    for part in ("baselines", "heads", "experts"):
        run = study / part
        c, architecture = read(run / "config.json"), read(run / "architecture.json")
        if c != condition_config(plan, base, part) or architecture != architecture_for(plan, part):
            raise AssertionError("A condition differs from its frozen plan")
        if digest(run / "manifest.json") != manifest["condition_manifest_sha256"][part]:
            raise AssertionError("A condition manifest changed")
        child_manifest = read(run / "manifest.json")
        if child_manifest["source_sha256"] != manifest["source_sha256"] or child_manifest["study_plan_sha256"] != manifest["plan_sha256"]:
            raise AssertionError("Conditions used different source or plans")
        for name, key in (("architecture.json", "architecture_sha256"), ("environment.json", "environment_sha256")):
            if digest(run / name) != child_manifest[key]:
                raise AssertionError(f"A condition input changed: {name}")
        for name, checksum in child_manifest["datasets_sha256"].items():
            if digest(run / name) != checksum:
                raise AssertionError("A frozen dataset changed")
        if digest(run / "artifacts.json") != summary["condition_artifacts_sha256"][part]:
            raise AssertionError("An artifact receipt changed")
        checks.append(verify(run) if part == "baselines" else verify_partition(run))
        child = read(run / "summary.json")
        if references is not None and child["linear_references"] != references:
            raise AssertionError("Linear references differ across conditions")
        references = child["linear_references"]
        for seed in plan["seeds"]:
            data_path = run / "data" / f"seed_{seed}.npz"
            fingerprint = dataset_fingerprint(data_path)
            if seed in fingerprints and fingerprint != fingerprints[seed]:
                raise AssertionError("Paired data or task order differs")
            fingerprints[seed] = fingerprint
            draws = np.load(run / f"seed_{seed}/online/batch_ids.npy", allow_pickle=False)
            if part != "baselines":
                original = np.load(study / "baselines" / f"seed_{seed}/online/batch_ids.npy", allow_pickle=False)
                if not np.array_equal(draws, original):
                    raise AssertionError("Current-only training batches differ across architectures")
                batch_comparisons += len(draws)
            else:
                with np.load(data_path, allow_pickle=False) as data:
                    weights = np.load(run / f"seed_{seed}/linear_weights.npy", allow_pickle=False)
                    scores = []
                    for task in range(c["tasks"]):
                        x = data[f"test_{task}_x"][:, :c["features"]]
                        x = np.column_stack([x, np.ones(len(x))])
                        scores.append(float(np.mean((x @ weights[task]).argmax(axis=1) == data[f"test_{task}_y"])))
                    reference = next(r for r in references if r["seed"] == seed)
                    if not np.array_equal(scores, reference["test_accuracy_by_task"]):
                        raise AssertionError("Saved linear weights do not reproduce the reference")
        stats = architecture_stats(c, architecture)
        for row in child["results"]:
            if row["parameter_count"] != stats["total_parameters"]:
                raise AssertionError("Recorded parameter count differs from architecture")
            if part == "baselines" and row["initial_model_sha256"] != model_digest(make_model(c, row["seed"])):
                raise AssertionError("Baseline initialization differs from the fixed seed")
            method = row["method"] if part == "baselines" else part
            matrix = np.asarray([s["test_accuracy_by_task"] for s in row["stages"]])
            order = np.array(row["order"])
            acquired = matrix[np.arange(len(order)), order]
            derived = [matrix[-1].mean(), acquired.mean(), (acquired[:-1] - matrix[-1, order[:-1]]).mean()]
            if not np.allclose(derived, [row[k] for k in METRICS], rtol=0, atol=1e-7):
                raise AssertionError("Summary metrics disagree with stage scores")
            results.append({
                "seed": row["seed"], "method": method,
                **{key: row[key] for key in (*METRICS, "updates", "update_examples", "parameter_count",
                    "training_seconds", "evaluation_seconds", "memory_maintenance_seconds", "retained_examples")},
                **stats, "training_dense_forward_macs": row["update_examples"] * stats["dense_forward_macs_per_example"],
                "evidence_seconds": row.get("evidence_seconds", 0),
            })
            for stage, scores in enumerate(matrix):
                for task, score in enumerate(scores):
                    stages.append({"seed": row["seed"], "method": method, "stage": stage,
                                   "learned_task": int(order[stage]), "evaluated_task": task, "test_accuracy": float(score)})
    decisions(results, plan)
    if len({(r["updates"], r["update_examples"]) for r in results}) != 1:
        raise AssertionError("Gradient update budgets differ")
    verification = {
        "status": "passed", "checkpoints_evaluated": sum(c["checkpoints_evaluated"] for c in checks),
        "training_example_draws_checked": sum(c["training_example_draws_checked"] for c in checks),
        "current_batch_comparisons": batch_comparisons,
        "allocation_batches_checked": audit_allocation(study / "baselines", condition_config(plan, base, "baselines")),
        "task_end_states_reproduced": sum(c.get("task_end_states_reproduced", 0) for c in checks),
        "inactive_component_checks": sum(c.get("inactive_component_checks", 0) for c in checks),
        "independent_prediction_checks": sum(c.get("independent_prediction_checks", 0) for c in checks),
        "data_and_task_order_paired": True, "current_only_batches_identical": True,
        "baseline_starting_weights_identical": True, "linear_references_reproduced": len(references),
    }
    if verification["checkpoints_evaluated"] != summary["neural_models"]:
        raise AssertionError("Verified model count differs")
    return plan, base, summary, results, stages, references, verification


def make_plot(output, results, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(LABELS)
    labels = ["Shared\ncurrent only", "Fixed 50%\nreplay", "Separate\noutputs", "Independent\nnetworks", "Full\nhistory"]
    colors = ["#7d858b", "#2274a5", "#a26713", "#23856d", "#7b63a5"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, key, title in zip(axes.flat[:3], METRICS,
                              ("Final test accuracy (%)", "New-task acquisition accuracy (%)", "Earlier accuracy lost (points)")):
        for i, method in enumerate(methods):
            values = np.array([r[key] * 100 for r in results if r["method"] == method])
            ax.bar(i, values.mean(), color=colors[i], alpha=0.65, width=0.65)
            ax.scatter(i + np.linspace(-0.18, 0.18, len(values)), values, color=colors[i], edgecolor="white", linewidth=0.4, s=25, zorder=3)
        ax.set_title(title)
        ax.set_xticks(range(5), labels)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        if key == METRICS[0]:
            ax.axhline(plan["final_accuracy_target"] * 100, color="#333333", linestyle=":", label="Final accuracy target")
            ax.legend(fontsize=8)
        if key != METRICS[2]:
            ax.set_ylim(0, 100)
        else:
            ax.axhline(0, color="#333333", linewidth=0.7)
    ax = axes[1, 1]
    values = [next(r for r in results if r["method"] == m)["dense_forward_macs_per_example"] for m in methods]
    ax.bar(range(5), np.array(values) / 1000, color=colors, alpha=0.65)
    ax.set_xticks(range(5), labels)
    ax.set_title("Dense multiply-add operations per example (thousands)")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.suptitle("Task partition study: fixed capacity and training updates", fontsize=16)
    fig.supxlabel("Dots show individual seeds. Operation counts cover a forward pass only.", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"results.{suffix}", dpi=160)
    plt.close(fig)


def report(study, output):
    plan, base, summary, results, stages, references, verification = verify_study(study)
    outcomes = decisions(results, plan)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("verification.json", verification), ("decisions.json", outcomes), ("results.json", results)):
        write_json(output / name, data)
    for name, rows in (("results.csv", results), ("stages.csv", stages)):
        with (output / name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    make_plot(output, results, plan)
    lines = ["# Task partition study", "", f"Study: {study.name}", ""]
    for method in ("heads", "experts", "replay"):
        outcome = outcomes[method]
        lines.extend([f"{LABELS[method]} {'passed' if outcome['passed'] else 'failed'} the fixed working criterion. {outcome['passing_count']} of {len(plan['seeds'])} seeds met all three targets; {outcome['required']} were required.", ""])
    lines.extend([
        f"The study trained {summary['neural_models']} neural models in {summary['wall_seconds']:.1f} seconds under resource limits. Peak process memory was {summary['peak_process_memory_mib']:.1f} MiB. Evidence verification passed.", "",
        f"The targets were at least {plan['final_accuracy_target']:.0%} final accuracy, at least {plan['retention_margin'] * 100:g} points less earlier accuracy lost than the shared current-only model, and at most {plan['acquisition_tolerance'] * 100:g} points lower acquisition accuracy. Acquisition is accuracy just after learning each task. All three conditions must hold for the same seed.", "",
        "Both architectures were fixed before scoring on fresh test seeds. There was no selection or parameter tuning on these scores. Earlier failed criteria remain unchanged.", "",
        "| Method | Final test accuracy | Acquisition accuracy | Earlier accuracy lost |",
        "| --- | ---: | ---: | ---: |",
    ])
    for method, label in LABELS.items():
        rows = [r for r in results if r["method"] == method]
        lines.append(f"| {label} | " + " | ".join(fmt([r[k] for r in rows], points=k == METRICS[2]) for k in METRICS) + " |")
    lines.extend(["", "Values show means and sample standard deviations across seeds. Negative earlier loss means improvement.", "",
                  f"The separate linear reference reached {fmt([r[METRICS[0]] for r in references])}. It has {references[0]['parameter_count']:,} fitted parameters and uses a different fitting procedure. Its training budget is not matched to the neural methods.", ""])
    for method in ("heads", "experts"):
        changes = {k: mean(r[k] for r in outcomes[method + "_minus_replay"]) * 100 for k in METRICS}
        lines.extend([f"Relative to fixed replay, {LABELS[method].lower()} changed final accuracy by {changes[METRICS[0]]:+.2f} points, acquisition by {changes[METRICS[1]]:+.2f} points, and earlier accuracy lost by {changes[METRICS[2]]:+.2f} points. These are descriptive paired differences. Positive loss change means more forgetting.", ""])
    lines.extend(["| Method | Final target | Retention target | Acquisition tolerance | All targets |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for method in ("replay", "heads", "experts"):
        outcome = outcomes[method]
        counts = outcome["target_counts"]
        lines.append(f"| {LABELS[method]} | {counts['final_accuracy']} | {counts['retention']} | {counts['acquisition']} | {outcome['passing_count']} |")
    lines.extend(["", "Individual outcomes follow. Verdicts use unrounded scores; displayed precision is only for readability.", ""])
    for method in ("heads", "experts", "replay"):
        lines.extend([LABELS[method] + ":", ""])
        for row in outcomes[method]["pairs"]:
            verdict = "passed" if row["passes"] else "failed: " + ", ".join(row["failed_targets"])
            lines.append(f"- Seed {row['seed']}: final {row['final_mean_accuracy']:.3%}; retention improvement {row['retention_improvement'] * 100:.3f} points; acquisition change {row['acquisition_change'] * 100:+.3f} points; {verdict}.")
        lines.append("")
    lines.extend([
        "All task components count toward the total parameter budget, including untrained ones. Active parameters are those used for one example. Multiply-add counts cover dense layers in one forward pass and exclude routing, activation, optimizer, and backward work.", "",
        "| Method | Total parameters | Active parameters | Forward multiply-adds per example | Retained examples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for method, label in LABELS.items():
        row = next(r for r in results if r["method"] == method)
        lines.append(f"| {label} | {row['total_parameters']:,} | {row['active_parameters_per_example']:,} | {row['dense_forward_macs_per_example']:,} | {row['retained_examples']:,} |")
    first = results[0]
    lines.extend(["", f"Every neural model used {first['updates']:,} updates and {first['update_examples']:,} training-example presentations. The new architectures used no replay or probes.", "",
                  "| Method | Training seconds | Evaluation seconds | Memory work seconds | Evidence seconds |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for method, label in LABELS.items():
        rows = [r for r in results if r["method"] == method]
        lines.append(f"| {label} | " + " | ".join(f"{mean(r[k] for r in rows):.3f}" for k in ("training_seconds", "evaluation_seconds", "memory_maintenance_seconds", "evidence_seconds")) + " |")
    lines.extend([
        "", "Times are per-model wall-clock means under throttling. Pauses and other machine load affect them. They are not CPU or energy measurements and cannot be compared directly with earlier unthrottled studies. Evidence timing covers the added component audits and stage snapshots; baseline evidence overhead is not separately measured.", "",
        f"Verification reproduced {verification['checkpoints_evaluated']} final checkpoints and {verification['task_end_states_reproduced']} task-end states. It audited {verification['training_example_draws_checked']:,} training-example draws, checked {verification['inactive_component_checks']} inactive components, and made {verification['independent_prediction_checks']} exact independent-task prediction comparisons.", "",
        f"The audit also compared {verification['current_batch_comparisons']:,} current-only batches across architectures and checked {verification['allocation_batches_checked']:,} baseline batch allocations. Saved linear weights reproduced all {verification['linear_references_reproduced']} references.", "",
        "Data and task order matched within each seed. Shared baselines started with identical weights. Different architectures used the same seed and layer initialization procedure, but their weight tensors differ. Inactive task parameters stayed unchanged. Independent networks preserved other tasks' predictions exactly, as their structure requires.", "",
        "The benchmark supplies task identity for routing. It uses six independent supervised tasks with simple linear rules. Fully separate networks cannot share learned features between tasks. Separate output layers still share changing hidden features. Neither approach in this study establishes transfer to new tasks or learned task recognition.", "",
        "This result does not establish language learning, unlimited continual learning, lower total learning cost, or a replacement for large-model pre-training. The success threshold is a working criterion, not a statistical significance test. Further independent tasks would require another capacity decision.", "",
        "The saved plan, source, data, checkpoints, batch IDs, component hashes, resource limits, and outcomes remain in the study directory. The launcher receipt separately records sampled resource use and any stop reason.", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))
    shutil.copy2(Path(__file__), output / "analysis_source.py")
    write_json(output / "provenance.json", {
        "study_summary_sha256": digest(study / "study_summary.json"),
        "analysis_source_sha256": digest(output / "analysis_source.py"),
        "dependency_sha256": {name: digest(ROOT / name) for name in DEPENDENCIES},
    })
    print("Report: " + str(output / "REPORT.md"), flush=True)
    print(json.dumps({m: outcomes[m]["passing_count"] for m in ("heads", "experts", "replay")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    report(study, args.output.resolve() if args.output else study / "report")


if __name__ == "__main__":
    main()
