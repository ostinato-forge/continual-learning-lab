"""Audit adaptive replay decisions and report the fixed fresh-seed comparison."""

import argparse
import csv
from pathlib import Path
import shutil
from statistics import mean

import numpy as np
import torch

from adaptive_controller import next_fraction
from adaptive_study import condition_config, validate_plan
from analyze_memory_sweep import center_spread, dataset_fingerprint, fmt, metrics_only, read
from analyze_replay_allocation import audit_allocation
from continual_lab import Examples, ReplayMemory, accuracy, digest, make_model, write_json
from memory_sweep import meets_target
from verify_run import verify


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = (
    "continual_lab.py", "run_experiment.py", "verify_run.py", "adaptive_controller.py",
    "adaptive_training.py", "adaptive_study.py", "analyze_memory_sweep.py",
    "analyze_replay_allocation.py", "memory_sweep.py", "replay_allocation.py",
)
METRICS = ("final_mean_accuracy", "mean_acquisition_accuracy", "old_task_retention_drop")
LABELS = {"online": "Current only", "replay": "Fixed 50% replay", "adaptive": "Adaptive replay", "full_history": "Full history"}


def audit_controller(run, c, policy):
    probe_calls = forward_examples = batches_checked = 0
    observations, schedules = [], []
    summary = read(run / "summary.json")
    for seed in c["seeds"]:
        folder = run / f"seed_{seed}" / "replay"
        trace = read(folder / "controller_trace.json")
        states = torch.load(folder / "probe_states.pt", map_location="cpu", weights_only=True)
        counts = np.load(folder / "replay_counts.npy", allow_pickle=False)
        draws = np.load(folder / "batch_ids.npy", allow_pickle=False)
        expected_shape = (c["tasks"] * c["steps_per_task"],)
        if counts.shape != expected_shape or len(trace) != len(states):
            raise AssertionError("Controller trace dimensions differ")
        memory = ReplayMemory(c["memory_capacity"], c["features"] + c["tasks"], seed + 400)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 500]))
        model = make_model(c, seed)
        position = 0
        seed_forwards = 0
        with np.load(run / "data" / f"seed_{seed}.npz", allow_pickle=False) as data:
            for stage, task in enumerate(data["order"]):
                current = Examples(
                    torch.from_numpy(data[f"train_{task}_x"]), torch.from_numpy(data[f"train_{task}_y"]),
                    data[f"train_{task}_ids"], np.full(c["train_examples"], task, dtype=np.int64), "train",
                )
                first = stage * c["steps_per_task"]
                block = draws[first:first + c["steps_per_task"]]
                actual_counts = c["batch_size"] - np.isin(block, current.ids).sum(axis=1)
                if not np.array_equal(actual_counts, counts[first:first + len(block)]):
                    raise AssertionError("Saved replay counts differ from actual batches")
                batches_checked += len(block)
                fraction = policy["initial_fraction"] if stage else 0.0
                reference = None
                if stage:
                    old = memory.examples()
                    indexes = rng.choice(len(old.ids), size=min(policy["probe_examples"], len(old.ids)), replace=False)
                    probe = Examples(old.x[indexes], old.y[indexes], old.ids[indexes], old.task_ids[indexes], "train")
                for step in range(c["steps_per_task"]):
                    if stage and step % policy["probe_interval"] == 0:
                        if position >= len(trace):
                            raise AssertionError("A required probe is missing")
                        row, state = trace[position], states[position]
                        if (row["stage"], row["step"]) != (stage, step) or (state["stage"], state["step"]) != (stage, step):
                            raise AssertionError("A probe ran at the wrong update")
                        if row["ids"] != probe.ids.tolist():
                            raise AssertionError("Probe examples differ from the frozen memory sampling rule")
                        model.load_state_dict(state["model"])
                        observed = accuracy(model, probe)
                        if abs(observed - row["accuracy"]) > 1e-7:
                            raise AssertionError("Probe checkpoint does not reproduce its recorded accuracy")
                        before = fraction
                        if reference is None:
                            reference = observed
                        else:
                            fraction = next_fraction(fraction, reference, observed, policy)
                        if (row["reference_accuracy"], row["fraction_before"], row["fraction_after"]) != (reference, before, fraction):
                            raise AssertionError("A controller decision differs from the fixed rule")
                        observations.append({"seed": seed, **{k: v for k, v in row.items() if k != "ids"}, "probe_examples": len(probe.ids)})
                        seed_forwards += len(probe.ids)
                        position += 1
                    if counts[first + step] != int(c["batch_size"] * fraction):
                        raise AssertionError("A training batch ignored the controller's chosen share")
                schedules.append({"seed": seed, "stage": stage, "mean_fraction": float(actual_counts.mean() / c["batch_size"])})
                memory.add(current)
        if position != len(trace):
            raise AssertionError("Unexpected extra controller probes")
        result = next(r for r in summary["results"] if r["seed"] == seed)
        accounting = result["controller"]
        expected = {
            "policy": policy, "probe_calls": position, "forward_examples": seed_forwards,
            "replayed_examples": int(counts.sum()),
            "later_batch_mean_fraction": float(counts[c["steps_per_task"]:].mean() / c["batch_size"]),
        }
        if any(accounting[key] != value for key, value in expected.items()):
            raise AssertionError("Controller cost or allocation accounting differs from the trace")
        probe_calls += position
        forward_examples += seed_forwards
    return {
        "probe_states_reproduced": probe_calls, "probe_forward_examples_checked": forward_examples,
        "adaptive_allocation_batches_checked": batches_checked,
        "observations": observations, "schedules": schedules,
    }


def verify_study(study):
    plan, base = read(study / "plan.json"), read(study / "base_config.json")
    validate_plan(plan, base)
    manifest, summary = read(study / "manifest.json"), read(study / "study_summary.json")
    if summary["status"] != "completed" or digest(study / "manifest.json") != summary["manifest_sha256"]:
        raise AssertionError("The study is incomplete or its manifest changed")
    for name, key in (("plan.json", "plan_sha256"), ("base_config.json", "base_sha256"), ("seed_audit.json", "seed_audit_sha256")):
        if digest(study / name) != manifest[key]:
            raise AssertionError(f"A frozen input changed: {name}")
    if set(read(study / "seed_audit.json")["previously_observed"]) & set(plan["seeds"]):
        raise AssertionError("Scored seeds were previously observed")
    for name, checksum in manifest["source_sha256"].items():
        if digest(study / "source" / name) != checksum:
            raise AssertionError("Frozen source changed")
    for name in DEPENDENCIES:
        if digest(ROOT / name) != manifest["source_sha256"][name]:
            raise AssertionError(f"Use the frozen experiment code: {name}")
    checks, results, fingerprints, initial_hashes = [], [], {}, {}
    references = None
    for part in ("baselines", "adaptive"):
        run = study / part
        c = read(run / "config.json")
        if c != condition_config(plan, base, part):
            raise AssertionError("A condition differs from the fixed settings")
        if digest(run / "manifest.json") != manifest["condition_manifest_sha256"][part]:
            raise AssertionError("A condition manifest changed")
        child_manifest = read(run / "manifest.json")
        if child_manifest["source_sha256"] != manifest["source_sha256"] or child_manifest["study_plan_sha256"] != manifest["plan_sha256"]:
            raise AssertionError("A condition used different source or a different plan")
        if part == "adaptive":
            if read(run / "policy.json") != plan["controller"] or digest(run / "policy.json") != child_manifest["policy_sha256"]:
                raise AssertionError("The controller policy changed")
        if digest(run / "artifacts.json") != summary["condition_artifacts_sha256"][part]:
            raise AssertionError("A condition's artifact receipt changed")
        checks.append(verify(run))
        child = read(run / "summary.json")
        references = child["linear_references"]
        for seed in plan["seeds"]:
            fingerprint = dataset_fingerprint(run / "data" / f"seed_{seed}.npz")
            if seed in fingerprints and fingerprints[seed] != fingerprint:
                raise AssertionError("Paired datasets differ")
            fingerprints[seed] = fingerprint
        for row in child["results"]:
            seed, initial = row["seed"], row["initial_model_sha256"]
            if seed in initial_hashes and initial_hashes[seed] != initial:
                raise AssertionError("Paired starting weights differ")
            initial_hashes[seed] = initial
            control = row.get("controller", {})
            results.append({
                "seed": seed, "method": "adaptive" if part == "adaptive" else row["method"],
                **metrics_only(row), **{key: row[key] for key in (
                    "training_seconds", "evaluation_seconds", "memory_maintenance_seconds", "updates", "update_examples", "parameter_count",
                )},
                "controller_seconds": control.get("controller_seconds", 0),
                "probe_forward_examples": control.get("forward_examples", 0),
                "probe_calls": control.get("probe_calls", 0), "evidence_seconds": control.get("evidence_seconds", 0),
                "later_batch_mean_fraction": control.get("later_batch_mean_fraction", plan["fixed_replay_fraction"] if row["method"] == "replay" else None),
            })
    control = audit_controller(study / "adaptive", condition_config(plan, base, "adaptive"), plan["controller"])
    fixed_batches = audit_allocation(study / "baselines", condition_config(plan, base, "baselines"))
    if len({r["parameter_count"] for r in results}) != 1 or len({(r["updates"], r["update_examples"]) for r in results}) != 1:
        raise AssertionError("Model sizes or gradient update budgets differ")
    verification = {
        "status": "passed", "checkpoints_evaluated": sum(r["checkpoints_evaluated"] for r in checks),
        "training_example_draws_checked": sum(r["training_example_draws_checked"] for r in checks),
        **{key: control[key] for key in ("probe_states_reproduced", "probe_forward_examples_checked")},
        "allocation_batches_checked": fixed_batches + control["adaptive_allocation_batches_checked"],
        "data_and_starting_weights_paired": True, "probe_access_limited_to_retained_training_data": True,
        "all_controller_decisions_reproduced": True,
    }
    if verification["checkpoints_evaluated"] != summary["neural_models"]:
        raise AssertionError("Verified model count differs")
    return plan, base, summary, results, references, control, verification


def decisions(results, plan):
    baseline = {row["seed"]: row for row in results if row["method"] == "online"}
    outcomes = {}
    for method in ("replay", "adaptive"):
        pairs = []
        for row in results:
            if row["method"] != method:
                continue
            old = baseline[row["seed"]]
            pairs.append({
                "seed": row["seed"], "final_mean_accuracy": row[METRICS[0]],
                "retention_improvement": old[METRICS[2]] - row[METRICS[2]],
                "acquisition_change": row[METRICS[1]] - old[METRICS[1]],
                "passes": meets_target(row, old, plan),
            })
        passed = sum(row["passes"] for row in pairs)
        outcomes[method] = {"passed": passed >= plan["required_successes"], "passing_count": passed, "required": plan["required_successes"], "pairs": pairs}
    fixed = {r["seed"]: r for r in results if r["method"] == "replay"}
    outcomes["adaptive_minus_fixed"] = [
        {"seed": r["seed"], **{key: r[key] - fixed[r["seed"]][key] for key in METRICS}}
        for r in results if r["method"] == "adaptive"
    ]
    return outcomes


def make_plot(output, results, control, plan, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "#fafaf8", "axes.facecolor": "#fafaf8", "savefig.facecolor": "#fafaf8"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, title in zip(axes.flat, METRICS, ("Final test accuracy", "Accuracy just after learning each new rule", "Earlier accuracy lost by the end")):
        for i, method in enumerate(LABELS):
            center, spread = center_spread([r[key] for r in results if r["method"] == method])
            ax.bar(i, center, yerr=spread, color="#26836f" if method == "adaptive" else "#7b92a5", capsize=4)
        ax.set_xticks(range(len(LABELS)), [v.replace(" ", "\n", 1) for v in LABELS.values()], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
        if key == METRICS[2]:
            ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:g}")
            ax.set_ylabel("Percentage points; lower is better")
        else:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if key == METRICS[0]:
            ax.axhline(plan["final_accuracy_target"], color="#b68c20", linestyle=":", label="Working target")
            ax.legend(fontsize=8, loc="lower right")
    ax = axes[1, 1]
    points = [center_spread([r["mean_fraction"] for r in control["schedules"] if r["stage"] == stage]) for stage in range(base["tasks"])]
    x = np.arange(1, base["tasks"] + 1)
    ax.errorbar(x, [p[0] for p in points], yerr=[p[1] for p in points], marker="o", color="#26836f", capsize=4, label="Adaptive replay")
    ax.plot(x, [0] + [plan["fixed_replay_fraction"]] * (base["tasks"] - 1), color="#7b92a5", linestyle="--", label="Fixed replay")
    ax.set_title("Actual replay share during each task")
    ax.set_xlabel("Task arrival number")
    ax.set_xticks(x)
    ax.set_ylabel("Old examples per training batch")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=9)
    fig.suptitle("Adaptive replay on fresh seeds", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.025, "Means and sample standard deviations. Gradient updates match; adaptive replay uses extra probe evaluations.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    fig.savefig(output / "results.png", dpi=170)
    fig.savefig(output / "results.svg")
    plt.close(fig)


def report(study, output):
    plan, base, summary, results, references, control, verification = verify_study(study)
    outcome = decisions(results, plan)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("verification.json", verification), ("decisions.json", outcome), ("results.json", results), ("controller_audit.json", control)):
        write_json(output / name, data)
    for name, rows in (("results.csv", results), ("controller.csv", control["observations"]), ("allocation.csv", control["schedules"])):
        with (output / name).open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    make_plot(output, results, control, plan, base)
    adaptive = outcome["adaptive"]
    lines = [
        "# Adaptive replay study", "", f"Study: {study.name}", "",
        f"Adaptive replay {'passed' if adaptive['passed'] else 'failed'} the fixed working criterion: {adaptive['passing_count']} of {len(plan['seeds'])} seeds met all targets, with {adaptive['required']} required. Fixed replay passed in {outcome['replay']['passing_count']} seeds.", "",
        f"The study trained {summary['neural_models']} neural models in {summary['wall_seconds']:.1f} seconds. Peak process memory was {summary['peak_process_memory_mib']:.1f} MiB. Evidence verification passed.", "",
        f"The combined target was at least {plan['final_accuracy_target']:.0%} final accuracy, at least {plan['retention_margin'] * 100:g} points less earlier accuracy lost than current-only training, and at most {plan['acquisition_tolerance'] * 100:g} points lower new-task acquisition accuracy. All three must hold in the same seed.", "",
        f"One controller was fixed before scoring. Fresh test seeds: {len(plan['seeds'])}. There was no parameter selection. Earlier studies and their negative results remain unchanged.", "",
        "| Method | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, label in LABELS.items():
        rows = [r for r in results if r["method"] == method]
        lines.append(f"| {label} | " + " | ".join(fmt([r[k] for r in rows], points=k == METRICS[2]) for k in METRICS) + " |")
    lines.extend(["", "Values show means and sample standard deviations across seeds. Earlier accuracy lost is signed; negative values mean improvement.", ""])
    changes = {key: mean(r[key] for r in outcome["adaptive_minus_fixed"]) * 100 for key in METRICS}
    lines.extend([
        f"Relative to fixed replay, adaptive replay changed mean final accuracy by {changes[METRICS[0]]:+.2f} points, acquisition by {changes[METRICS[1]]:+.2f} points, and earlier accuracy lost by {changes[METRICS[2]]:+.2f} points. Positive loss change means more forgetting. These paired differences are descriptive, not significance tests.", "",
        f"The separate linear reference reached {fmt([r[METRICS[0]] for r in references])}. It uses a different fitting procedure and a model suited to these rules.", "",
        "Individual adaptive outcomes:", "",
    ])
    for row in adaptive["pairs"]:
        lines.append(f"- Seed {row['seed']}: final accuracy {row['final_mean_accuracy']:.1%}; retention improvement {row['retention_improvement'] * 100:.1f} points; acquisition change {row['acquisition_change'] * 100:+.1f} points; {'passed' if row['passes'] else 'failed'}.")
    baseline = {r["seed"]: r for r in results if r["method"] == "online"}
    lines.extend(["", "Separate target counts explain the outcome. Acceptance still requires all three together.", "", "| Method | Final target | Retention target | Acquisition tolerance |", "| --- | ---: | ---: | ---: |"])
    for method in ("replay", "adaptive"):
        rows = [r for r in results if r["method"] == method]
        counts = [sum(r[METRICS[0]] >= plan["final_accuracy_target"] for r in rows), sum(baseline[r["seed"]][METRICS[2]] - r[METRICS[2]] >= plan["retention_margin"] for r in rows), sum(r[METRICS[1]] - baseline[r["seed"]][METRICS[1]] >= -plan["acquisition_tolerance"] for r in rows)]
        lines.append(f"| {LABELS[method]} | {counts[0]} | {counts[1]} | {counts[2]} |")
    adaptive_rows = [r for r in results if r["method"] == "adaptive"]
    lines.extend([
        "", "## Compute and controller behavior", "",
        f"Every neural model used {base['tasks'] * base['steps_per_task']:,} updates and {base['tasks'] * base['steps_per_task'] * base['batch_size']:,} training-example presentations with forward and backward passes. Adaptive replay added {adaptive_rows[0]['probe_calls']} probe calls and {adaptive_rows[0]['probe_forward_examples']:,} forward-only example evaluations per seed. Probe evaluations made no gradient updates.", "",
        "| Method | Mean training seconds | Mean controller seconds | Mean benchmark evaluation seconds | Mean memory work seconds | Mean probe evidence capture seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method, label in LABELS.items():
        rows = [r for r in results if r["method"] == method]
        lines.append(f"| {label} | " + " | ".join(f"{mean(r[key] for r in rows):.3f}" for key in ("training_seconds", "controller_seconds", "evaluation_seconds", "memory_maintenance_seconds", "evidence_seconds")) + " |")
    ups = sum(r["fraction_after"] > r["fraction_before"] for r in control["observations"])
    downs = sum(r["fraction_after"] < r["fraction_before"] for r in control["observations"])
    lines.extend([
        "", f"The mean replay share after the first task was {mean(r['later_batch_mean_fraction'] for r in adaptive_rows):.1%}. Across all seeds, the controller increased the share {ups} times and decreased it {downs} times. It reset to the initial share at each task.", "",
        "The experiment matches gradient updates, not total compute. Extra probes are charged to the adaptive learner. Evidence capture supports auditing and is reported separately. Benchmark validation and test scores never control training. Wall times depend on system load; these results do not establish lower energy use or cheaper learning.", "",
        "## Verification and limits", "",
        f"Verification reproduced {verification['checkpoints_evaluated']} final models and {verification['probe_states_reproduced']} probe model states. It checked {verification['training_example_draws_checked']:,} training-example draws, {verification['probe_forward_examples_checked']:,} probe evaluations, and {verification['allocation_batches_checked']:,} batch allocations.", "",
        "Probe IDs matched the frozen sampling rule and belonged to retained training memory. Recorded predictions and threshold decisions matched saved model states. Data, task order, model size, and initial weights matched within each seed.", "",
        "Training-memory accuracy can improve without improved generalization. The test scores measure unseen examples. These remain supervised synthetic tasks with supplied identities and linear boundaries. The result does not establish language learning, general concepts, or replacement of large-model pre-training.", "",
        "The model does not discover new architectures or choose its own goals. It changes one training allocation using a rule supplied by the experimenter. The required seed count is a working criterion, not a statistical significance test.", "",
        "The parent directory retains the fixed plan, source, data, model states, training draws, controller trace, and per-method metrics. CSV files contain every scored seed and controller observation.", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))
    shutil.copy2(Path(__file__), output / "analysis_source.py")
    write_json(output / "provenance.json", {
        "study_summary_sha256": digest(study / "study_summary.json"),
        "analysis_source_sha256": digest(output / "analysis_source.py"),
        "dependency_sha256": {name: digest(ROOT / name) for name in DEPENDENCIES},
    })
    print("Report: " + str(output / "REPORT.md"))
    print({"adaptive_successes": adaptive["passing_count"], "fixed_successes": outcome["replay"]["passing_count"], "required": adaptive["required"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    report(study, args.output.resolve() if args.output else study / "report")


if __name__ == "__main__":
    main()
