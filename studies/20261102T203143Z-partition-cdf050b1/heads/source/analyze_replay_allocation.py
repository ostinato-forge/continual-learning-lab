"""Select a replay share on validation data, then check fresh test results."""

import argparse
import csv
from pathlib import Path
import shutil
from statistics import mean

import numpy as np

from analyze_memory_sweep import (
    center_spread, dataset_fingerprint, fmt, metrics_only, read, validation_metrics,
)
from continual_lab import digest, write_json
from memory_sweep import meets_target
from replay_allocation import condition_config, select_fraction, validate_plan
from verify_run import verify


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = (
    "continual_lab.py", "run_experiment.py", "verify_run.py", "memory_sweep.py",
    "analyze_memory_sweep.py", "replay_allocation.py",
)
METRICS = ("final_mean_accuracy", "mean_acquisition_accuracy", "old_task_retention_drop")


def check_study_inputs(study, selection=False):
    plan, base = read(study / "plan.json"), read(study / "base_config.json")
    validate_plan(plan, base)
    manifest = read(study / "manifest.json")
    for name, key in (("plan.json", "plan_sha256"), ("base_config.json", "base_sha256")):
        if digest(study / name) != manifest[key]:
            raise AssertionError(f"Frozen input changed: {name}")
    for name, expected in manifest["source_sha256"].items():
        if digest(study / "source" / name) != expected:
            raise AssertionError(f"Frozen source changed: {name}")
    for name, expected in manifest["config_sha256"].items():
        if digest(study / name) != expected:
            raise AssertionError(f"Frozen condition changed: {name}")
    names = DEPENDENCIES + ((Path(__file__).name,) if selection else ())
    for name in names:
        if digest(ROOT / name) != manifest["source_sha256"][name]:
            raise AssertionError(f"Use the frozen experiment code: {name}")
    return plan, base, manifest


def audit_allocation(run, c):
    """Count old examples in every saved replay batch, including the first task."""
    checked = 0
    for seed in c["seeds"]:
        with np.load(run / "data" / f"seed_{seed}.npz", allow_pickle=False) as data:
            for method in c["methods"]:
                if method == "full_history":
                    continue
                draws = np.load(run / f"seed_{seed}" / method / "batch_ids.npy", allow_pickle=False)
                for stage, task in enumerate(data["order"]):
                    block = draws[stage * c["steps_per_task"]:(stage + 1) * c["steps_per_task"]]
                    current_counts = np.isin(block, data[f"train_{task}_ids"]).sum(axis=1)
                    old_count = int(c["batch_size"] * c["replay_fraction"]) if method == "replay" and stage else 0
                    if not np.all(current_counts == c["batch_size"] - old_count):
                        raise AssertionError("A batch did not use the planned replay share")
                    checked += len(block)
    return checked


def load_condition(study, receipt, expected_config, manifest):
    run = study / receipt["run"]
    if digest(run / "artifacts.json") != receipt["artifacts_sha256"]:
        raise AssertionError("A condition receipt changed")
    if read(run / "config.json") != expected_config:
        raise AssertionError("A condition differs from its declared settings")
    child_manifest = read(run / "manifest.json")
    for name in DEPENDENCIES:
        if child_manifest["source_sha256"][name] != manifest["source_sha256"][name]:
            raise AssertionError(f"Code differs between conditions: {name}")
    checked = verify(run)
    checked["allocation_batches_checked"] = audit_allocation(run, expected_config)
    child = read(run / "summary.json")
    if receipt["model_count"] != len(child["results"]):
        raise AssertionError("The model count differs from its receipt")
    return run, child, checked


def check_pairs(run, c, child, fingerprints, initial_hashes):
    for seed in c["seeds"]:
        fingerprint = dataset_fingerprint(run / "data" / f"seed_{seed}.npz")
        if seed in fingerprints and fingerprints[seed] != fingerprint:
            raise AssertionError("Data or task order differs between replay shares")
        fingerprints[seed] = fingerprint
    for row in child["results"]:
        seed, initial = row["seed"], row["initial_model_sha256"]
        if seed in initial_hashes and initial_hashes[seed] != initial:
            raise AssertionError("Initial weights differ between replay shares")
        initial_hashes[seed] = initial


def collect_selection(study):
    plan, base, manifest = check_study_inputs(study, selection=True)
    receipts = read(study / "exploration.json")
    if [row["fraction"] for row in receipts] != plan["replay_fractions"]:
        raise AssertionError("The exploration conditions differ from the plan")
    fingerprints, initial_hashes, anchors, references = {}, {}, {}, {}
    cases, verification = [], []
    for receipt in receipts:
        fraction = receipt["fraction"]
        c = condition_config(plan, base, fraction)
        run, child, checked = load_condition(study, receipt, c, manifest)
        check_pairs(run, c, child, fingerprints, initial_hashes)
        verification.append({"fraction": fraction, **checked})
        anchors[receipt["run"]] = receipt["artifacts_sha256"]
        if fraction == 0:
            references["full_history"] = [
                {"seed": row["seed"], **metrics_only(row)}
                for row in child["results"] if row["method"] == "full_history"
            ]
            references["linear"] = child["linear_references"]
        for row in child["results"]:
            if row["method"] == "full_history":
                continue
            cases.append({
                "fraction": fraction, "seed": row["seed"], "method": row["method"],
                "validation": validation_metrics(run, row, c), "test": metrics_only(row),
                **{key: row[key] for key in (
                    "training_seconds", "evaluation_seconds", "memory_maintenance_seconds",
                    "retained_examples", "parameter_count", "updates", "update_examples",
                )},
            })
    selected = select_fraction(cases, plan)
    selected.update({
        "selection_basis": "Validation metrics only. Test scores do not enter the selection function.",
        "cases": cases, "references": references, "verification": verification,
        "run_artifact_hashes": anchors, "dataset_fingerprints": fingerprints,
        "paired_initial_hashes": initial_hashes,
    })
    write_json(study / "selection.json", selected)
    print({"chosen_fraction": selected["chosen_fraction"], "choices": selected["choices"]})
    return selected


def confirmation_result(study, summary, plan, base, manifest):
    chosen = summary["chosen_fraction"]
    receipt, reference = summary["confirmation"], summary["paired_reference"]
    if chosen is None:
        if receipt is not None or reference is not None:
            raise AssertionError("Confirmation ran without a selected share")
        return {"status": "not_run", "reason": "No replay share met the validation target"}
    if receipt is None or (chosen != plan["reference_fraction"] and reference is None):
        raise AssertionError("A declared confirmation condition is missing")
    if chosen == plan["reference_fraction"] and reference is not None:
        raise AssertionError("The same reference was run twice")
    c = condition_config(plan, base, chosen, "confirmation")
    run, child, checked = load_condition(study, receipt, c, manifest)
    fingerprints, initial_hashes = {}, {}
    check_pairs(run, c, child, fingerprints, initial_hashes)
    checks = [checked]
    results = [
        {"seed": row["seed"], "method": row["method"], **metrics_only(row),
         "training_seconds": row["training_seconds"]}
        for row in child["results"]
    ]
    baseline = {row["seed"]: row for row in results if row["method"] == "online"}
    selected = {row["seed"]: row for row in results if row["method"] == "replay"}
    if reference is not None:
        c = condition_config(plan, base, plan["reference_fraction"], "reference")
        run, other, checked = load_condition(study, reference, c, manifest)
        check_pairs(run, c, other, fingerprints, initial_hashes)
        checks.append(checked)
        reference_rows = {
            row["seed"]: {"seed": row["seed"], "method": "reference_replay", **metrics_only(row),
                          "training_seconds": row["training_seconds"]}
            for row in other["results"]
        }
        results.extend(reference_rows.values())
    else:
        reference_rows = selected
    pairs, reference_pairs = [], []
    for seed in plan["confirmation_seeds"]:
        row, old, other = selected[seed], baseline[seed], reference_rows[seed]
        pairs.append({
            "seed": seed, "final_mean_accuracy": row["final_mean_accuracy"],
            "retention_improvement": old["old_task_retention_drop"] - row["old_task_retention_drop"],
            "acquisition_change": row["mean_acquisition_accuracy"] - old["mean_acquisition_accuracy"],
            "passes": meets_target(row, old, plan),
        })
        reference_pairs.append({"seed": seed, **{key + "_change": row[key] - other[key] for key in METRICS}})
    passed = sum(row["passes"] for row in pairs)
    return {
        "status": "completed", "passed": passed >= plan["required_confirmation_successes"],
        "passing_count": passed, "required": plan["required_confirmation_successes"],
        "pairs": pairs, "verification": checks, "results": results,
        "reference_pairs": reference_pairs, "reference_differs_from_selected": reference is not None,
        "dataset_fingerprints": fingerprints, "paired_initial_hashes": initial_hashes,
        "linear_references": child["linear_references"],
    }


def validation_breakdown(cases, plan):
    baseline = {row["seed"]: row["validation"] for row in cases if row["fraction"] == 0}
    counts = []
    for fraction in plan["replay_fractions"][1:]:
        rows = [row for row in cases if row["fraction"] == fraction]
        counts.append({
            "fraction": fraction,
            "final": sum(row["validation"][METRICS[0]] >= plan["final_accuracy_target"] for row in rows),
            "retention": sum(baseline[row["seed"]][METRICS[2]] - row["validation"][METRICS[2]] >= plan["retention_margin"] for row in rows),
            "acquisition": sum(row["validation"][METRICS[1]] - baseline[row["seed"]][METRICS[1]] >= -plan["acquisition_tolerance"] for row in rows),
        })
    return counts


def method_labels(chosen, plan):
    labels = {"online": "Current examples only", "replay": f"Replay {chosen:.1%}", "full_history": "Full history"}
    if chosen != plan["reference_fraction"]:
        labels["reference_replay"] = f"Replay {plan['reference_fraction']:.1%}"
    return labels


def make_plot(output, plan, selection, confirmation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "#fafaf8",
        "axes.facecolor": "#fafaf8", "savefig.facecolor": "#fafaf8",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ("Final accuracy on unfamiliar examples", "Accuracy just after learning each new rule", "Earlier accuracy lost by the end")
    for ax, key, title in zip(axes.flat, METRICS, titles):
        points = [center_spread([r["test"][key] for r in selection["cases"] if r["fraction"] == v]) for v in plan["replay_fractions"]]
        ax.errorbar(plan["replay_fractions"], [p[0] for p in points], yerr=[p[1] for p in points], color="#26836f", marker="o", capsize=4)
        ax.axhline(mean(r[key] for r in selection["references"]["full_history"]), color="#3274a1", linestyle="--", label="Full history mean")
        if key == METRICS[0]:
            ax.axhline(plan["final_accuracy_target"], color="#b68c20", linestyle=":", label="Working target")
            ax.legend(fontsize=8, loc="lower right")
        if selection["chosen_fraction"] is not None:
            ax.axvline(selection["chosen_fraction"], color="#777777", linewidth=1, alpha=0.5)
        ax.set_xticks(plan["replay_fractions"], [f"{v:.1%}" for v in plan["replay_fractions"]])
        ax.set_xlabel("Share of each later batch drawn from old examples")
        ax.set_title(title)
        ax.set_ylim(-0.05 if key == METRICS[2] else 0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if key == METRICS[2]:
            ax.set_ylabel("Accuracy difference (percentage points)")
            ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}")
        ax.grid(axis="y", alpha=0.15)
    ax = axes[1, 1]
    if confirmation["status"] == "completed":
        labels = method_labels(selection["chosen_fraction"], plan)
        for i, (method, label) in enumerate(labels.items()):
            average, spread = center_spread([r[METRICS[0]] for r in confirmation["results"] if r["method"] == method])
            ax.bar(i, average, yerr=spread, color="#26836f" if method == "replay" else "#7b92a5", capsize=4)
        ax.set_xticks(range(len(labels)), [v.replace(" ", "\n", 1) for v in labels.values()], fontsize=9)
        ax.set_title(f"Fresh confirmation: {confirmation['passing_count']} of {len(plan['confirmation_seeds'])} met all targets")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Final test accuracy")
    else:
        ax.axis("off")
        ax.text(0.5, 0.55, "No replay share qualified\nConfirmation was skipped", ha="center", va="center", fontsize=14)
        ax.text(0.5, 0.3, f"All targets were required in {plan['required_exploration_successes']} of {len(plan['exploration_seeds'])} validation seeds.", ha="center", fontsize=10)
    fig.suptitle(f"Replay allocation with {plan['memory_capacity']:,} stored examples", fontsize=16, fontweight="bold")
    marker = " The vertical line marks the selected share." if selection["chosen_fraction"] is not None else ""
    fig.text(0.5, 0.025, "Means and sample standard deviations across seeds. All neural models use the same training budget." + marker, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    fig.savefig(output / "results.png", dpi=170)
    fig.savefig(output / "results.svg")
    plt.close(fig)


def report(study, output):
    plan, base, manifest = check_study_inputs(study)
    summary, selection = read(study / "study_summary.json"), read(study / "selection.json")
    if summary["status"] != "completed":
        raise AssertionError("The study is incomplete")
    for name, key in (("selection.json", "selection_sha256"), ("manifest.json", "manifest_sha256")):
        if digest(study / name) != summary[key]:
            raise AssertionError(f"Study receipt changed: {name}")
    if read(study / "exploration.json") != summary["exploration"]:
        raise AssertionError("Exploration receipts changed")
    for relative, expected in selection["run_artifact_hashes"].items():
        run = study / relative
        if digest(run / "artifacts.json") != expected:
            raise AssertionError("An exploration receipt changed")
        for artifact, checksum in read(run / "artifacts.json").items():
            if digest(run / artifact) != checksum:
                raise AssertionError(f"An exploration artifact changed: {artifact}")
    recomputed = select_fraction(selection["cases"], plan)
    if any(recomputed[key] != selection[key] for key in recomputed) or summary["chosen_fraction"] != selection["chosen_fraction"]:
        raise AssertionError("Selection does not follow the declared rule")
    confirmation = confirmation_result(study, summary, plan, base, manifest)
    checks = selection["verification"] + confirmation.get("verification", [])
    verification = {
        "status": "passed",
        **{key: sum(r[key] for r in checks) for key in (
            "checkpoints_evaluated", "training_example_draws_checked", "allocation_batches_checked",
        )},
        "data_and_initial_weights_paired_across_fractions": True,
        "selection_uses_validation_only": True,
        "confirmation_seeds_disjoint_from_prior_runs_and_exploration": True,
        "saved_inputs_and_outputs_unchanged": True,
    }
    if verification["checkpoints_evaluated"] != summary["neural_models"]:
        raise AssertionError("Verified model count differs from the study total")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "confirmation.json", confirmation)
    write_json(output / "verification.json", verification)
    breakdown = validation_breakdown(selection["cases"], plan)
    write_json(output / "validation_breakdown.json", breakdown)
    fields = ["fraction", "seed", "split", *METRICS, "training_seconds", "evaluation_seconds", "memory_maintenance_seconds", "updates", "update_examples", "retained_examples"]
    with (output / "exploration.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in selection["cases"]:
            for split in ("validation", "test"):
                writer.writerow({**row, **row[split], "split": split})
    if confirmation["status"] == "completed":
        with (output / "confirmation.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "method", *METRICS, "training_seconds"])
            writer.writeheader()
            writer.writerows(confirmation["results"])
    make_plot(output, plan, selection, confirmation)
    chosen = selection["chosen_fraction"]
    lines = [
        "# Replay allocation study", "", f"Study: {study.name}", "",
        f"Completed in {summary['wall_seconds']:.1f} seconds. Trained {summary['neural_models']} neural models. Evidence verification passed.", "",
        (f"The smallest tested share meeting the validation target was {chosen:.1%}." if chosen is not None else
         "No tested replay share met the fixed validation target. Confirmation was skipped."), "",
        f"The rule requires at least {plan['final_accuracy_target']:.0%} final accuracy, at least {plan['retention_margin'] * 100:g} percentage points less earlier accuracy lost than current-only training, and at most {plan['acquisition_tolerance'] * 100:g} points lower new-task acquisition accuracy. All three must hold in the same seed.", "",
        f"Selection required {plan['required_exploration_successes']} of {len(plan['exploration_seeds'])} validation seeds. Confirmation requires {plan['required_confirmation_successes']} of {len(plan['confirmation_seeds'])} fresh test seeds. No threshold was changed after scoring.", "",
    ]
    if confirmation["status"] == "completed":
        lines.extend([
            "## Fresh confirmation", "",
            f"The selected share {'passed' if confirmation['passed'] else 'failed'} confirmation: {confirmation['passing_count']} of {len(plan['confirmation_seeds'])} seeds met all targets.", "",
            "| Method | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost |",
            "| --- | ---: | ---: | ---: |",
        ])
        for method, label in method_labels(chosen, plan).items():
            rows = [r for r in confirmation["results"] if r["method"] == method]
            lines.append(f"| {label} | " + " | ".join(fmt([r[key] for r in rows], points=key == METRICS[2]) for key in METRICS) + " |")
        lines.extend(["", "Individual confirmation outcomes:", ""])
        for row in confirmation["pairs"]:
            lines.append(f"- Seed {row['seed']}: final accuracy {row['final_mean_accuracy']:.1%}; retention improvement {row['retention_improvement'] * 100:.1f} points; acquisition change {row['acquisition_change'] * 100:+.1f} points; {'passed' if row['passes'] else 'failed'}.")
        lines.append("")
        if confirmation["reference_differs_from_selected"]:
            changes = {key: mean(r[key + "_change"] for r in confirmation["reference_pairs"]) * 100 for key in METRICS}
            lines.extend([
                f"Against the paired {plan['reference_fraction']:.1%} replay reference, the selected share changed mean acquisition accuracy by {changes[METRICS[1]]:+.2f} points, final accuracy by {changes[METRICS[0]]:+.2f} points, and earlier accuracy lost by {changes[METRICS[2]]:+.2f} points. Positive loss change means more forgetting.", "",
                "These paired differences are descriptive. They are not a separate acceptance test or evidence of lower compute cost.", "",
            ])
        else:
            lines.extend(["The selected share equals the reference share, so confirmation does not test a lower replay allocation.", ""])
    lines.extend([
        "## Exploration", "",
        "Test scores below describe the fixed comparison. Selection used validation scores only.", "",
        "| Replay share | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost | Validation successes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    choices = {row["fraction"]: row for row in selection["choices"]}
    for fraction in plan["replay_fractions"]:
        rows = [row["test"] for row in selection["cases"] if row["fraction"] == fraction]
        count = f"{choices[fraction]['passing_count']} of {len(plan['exploration_seeds'])}" if fraction else "reference"
        lines.append(f"| {fraction:.1%} | " + " | ".join(fmt([r[key] for r in rows], points=key == METRICS[2]) for key in METRICS) + f" | {count} |")
    lines.extend([
        "", "Values show means and sample standard deviations across seeds. Earlier accuracy lost is signed; negative values mean improvement.", "",
        "Each validation condition counted separately:", "",
        "| Replay share | Final accuracy target passed | Retention target passed | Acquisition tolerance passed |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in breakdown:
        lines.append(f"| {row['fraction']:.1%} | {row['final']} | {row['retention']} | {row['acquisition']} |")
    lines.extend([
        "", "These diagnostic counts do not change the rule. All three conditions must pass together.", "",
        f"The exploration full-history reference reached {fmt([r[METRICS[0]] for r in selection['references']['full_history']])}. The separate linear reference reached {fmt([r[METRICS[0]] for r in selection['references']['linear']])}.", "",
        "## Cost and verification", "",
        f"Replay memory held up to {plan['memory_capacity']:,} examples. Every neural model used {base['tasks'] * base['steps_per_task']:,} updates and {base['tasks'] * base['steps_per_task'] * base['batch_size']:,} example presentations. Architecture, optimizer, and task distribution were fixed. The first task uses current examples only because no earlier memory exists.", "",
        f"Verification reproduced {verification['checkpoints_evaluated']} final checkpoints, audited {verification['training_example_draws_checked']:,} training-example draws, and checked exact old/current counts in {verification['allocation_batches_checked']:,} batches. Data, task order, and starting weights matched across conditions within each seed.", "",
        "Study time includes training, process startup, and validation selection. Confirmation readback and reporting occur afterward. Per-model training times are saved in CSV files. Equal update budgets do not establish equal energy use.", "",
        "Memory size counts retained training examples. It excludes model weights, optimizer state, the current training block, and evaluation data. The benchmark saves full datasets for auditing; each learner's permitted access is checked from its batch trace.", "",
        "## Limits", "",
        "The selected share, if any, is only the smallest tested share meeting this target. The seed thresholds are experimental rules, not statistical significance tests. A different task family may give different results.", "",
        "These are supervised synthetic tasks with supplied machine identities and linear rule boundaries. The separate linear reference uses a different procedure and a model suited to the rules. This experiment does not establish language ability, autonomous exploration, general concept learning, or replacement of language-model pre-training.", "",
        "Earlier negative findings remain unchanged. This study changes the replay allocation on fresh seeds. It does not revise the earlier memory-size study's failed criterion.", "",
        "## Files", "",
        "- results.png and results.svg contain the charts.",
        "- exploration.csv contains every exploration seed's validation and test metrics.",
        "- confirmation.json contains individual outcomes and paired reference differences, or the reason confirmation was skipped.",
        "- verification.json and validation_breakdown.json contain audit counts and diagnostic counts.",
        "- The parent directory retains the frozen plan, source, data, selection, batch traces, and checkpoints.", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))
    shutil.copy2(Path(__file__), output / "analysis_source.py")
    write_json(output / "provenance.json", {
        "study_summary_sha256": digest(study / "study_summary.json"),
        "selection_sha256": digest(study / "selection.json"),
        "analysis_source_sha256": digest(output / "analysis_source.py"),
        "dependency_sha256": {name: digest(ROOT / name) for name in DEPENDENCIES},
    })
    print("Report: " + str(output / "REPORT.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "report"))
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    if args.action == "select":
        collect_selection(study)
    else:
        report(study, args.output.resolve() if args.output else study / "report")


if __name__ == "__main__":
    main()
