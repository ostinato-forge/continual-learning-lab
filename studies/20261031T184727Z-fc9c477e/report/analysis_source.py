"""Select a memory size using validation data and report fresh confirmation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean, stdev

import numpy as np
import torch

from continual_lab import Examples, accuracy, digest, make_model, write_json
from memory_sweep import condition_config, meets_target, select_capacity, validate_plan
from verify_run import verify


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def check_study_inputs(study):
    plan = read(study / "plan.json")
    base = read(study / "base_config.json")
    validate_plan(plan, base)
    manifest = read(study / "manifest.json")
    if digest(study / "plan.json") != manifest["plan_sha256"]:
        raise AssertionError("The study plan changed")
    if digest(study / "base_config.json") != manifest["base_sha256"]:
        raise AssertionError("The base configuration changed")
    for name, expected in manifest["source_sha256"].items():
        if digest(study / "source" / name) != expected:
            raise AssertionError(f"The frozen source changed: {name}")
    for name, expected in manifest["config_sha256"].items():
        if digest(study / name) != expected:
            raise AssertionError(f"A frozen condition changed: {name}")
    for name in ("continual_lab.py", "run_experiment.py", "verify_run.py"):
        if digest(ROOT / name) != manifest["source_sha256"][name]:
            raise AssertionError(f"Use the frozen learning code to analyze this study: {name}")
    return plan, base, manifest


def dataset_fingerprint(path):
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            array = data[name]
            h.update(name.encode())
            h.update(str(array.shape).encode())
            h.update(str(array.dtype).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def metrics_only(result):
    return {key: result[key] for key in (
        "final_mean_accuracy", "mean_acquisition_accuracy", "old_task_retention_drop",
    )}


def validation_metrics(run, result, c):
    checkpoint = torch.load(
        run / f"seed_{result['seed']}" / result["method"] / "checkpoint.pt",
        map_location="cpu", weights_only=True,
    )
    model = make_model(c, result["seed"])
    model.load_state_dict(checkpoint["model"])
    scores = []
    with np.load(run / "data" / f"seed_{result['seed']}.npz", allow_pickle=False) as data:
        for task in range(c["tasks"]):
            examples = Examples(
                torch.from_numpy(data[f"validation_{task}_x"]),
                torch.from_numpy(data[f"validation_{task}_y"]),
                data[f"validation_{task}_ids"],
                np.full(c["validation_examples"], task, dtype=np.int64), "validation",
            )
            scores.append(accuracy(model, examples))
    acquired = [
        next(row["validation_accuracy"] for row in result["curves"]
             if row["stage"] == stage and row["step"] == c["steps_per_task"])
        for stage in range(c["tasks"])
    ]
    return {
        "final_mean_accuracy": float(np.mean(scores)),
        "mean_acquisition_accuracy": float(np.mean(acquired)),
        "old_task_retention_drop": float(np.mean([
            acquired[stage] - scores[task]
            for stage, task in enumerate(result["order"][:-1])
        ])),
        "final_accuracy_by_task": scores,
    }


def collect_selection(study):
    plan, base, manifest = check_study_inputs(study)
    receipts = read(study / "exploration.json")
    if [row["capacity"] for row in receipts] != plan["capacities"]:
        raise AssertionError("The exploration conditions differ from the frozen plan")
    fingerprints = {}
    initial_hashes = {}
    cases = []
    verification = []
    references = {}
    anchors = {}
    for receipt in receipts:
        capacity = receipt["capacity"]
        run = study / receipt["run"]
        if digest(run / "artifacts.json") != receipt["artifacts_sha256"]:
            raise AssertionError("Condition artifact receipt changed")
        c = read(run / "config.json")
        if c != condition_config(plan, base, capacity):
            raise AssertionError("A condition changed more than its planned settings")
        child_manifest = read(run / "manifest.json")
        for name in ("continual_lab.py", "run_experiment.py", "verify_run.py"):
            if child_manifest["source_sha256"][name] != manifest["source_sha256"][name]:
                raise AssertionError("Learning code differs between conditions")
        checked = verify(run)
        verification.append({"capacity": capacity, **checked})
        anchors[receipt["run"]] = receipt["artifacts_sha256"]
        summary = read(run / "summary.json")
        if capacity == 0:
            references["full_history"] = [
                {"seed": row["seed"], **metrics_only(row), "training_seconds": row["training_seconds"]}
                for row in summary["results"] if row["method"] == "full_history"
            ]
            references["linear"] = summary["linear_references"]
        for seed in plan["exploration_seeds"]:
            fingerprint = dataset_fingerprint(run / "data" / f"seed_{seed}.npz")
            if seed in fingerprints and fingerprints[seed] != fingerprint:
                raise AssertionError("Data or task order differs between capacities")
            fingerprints[seed] = fingerprint
        for result in summary["results"]:
            seed = result["seed"]
            initial = result["initial_model_sha256"]
            if seed in initial_hashes and initial_hashes[seed] != initial:
                raise AssertionError("Initial weights differ between capacities")
            initial_hashes[seed] = initial
            if result["method"] == "full_history":
                continue
            cases.append({
                "capacity": capacity, "seed": seed, "method": result["method"],
                "validation": validation_metrics(run, result, c),
                "test": metrics_only(result),
                "training_seconds": result["training_seconds"],
                "retained_examples": result["retained_examples"],
                "parameter_count": result["parameter_count"],
                "updates": result["updates"],
                "update_examples": result["update_examples"],
            })
    selected = select_capacity(cases, plan)
    baseline = {row["seed"]: row["test"] for row in cases if row["capacity"] == 0}
    replica = [row for row in cases if row["capacity"] == plan["replication_capacity"]]
    replication_passes = [
        row["seed"] for row in replica
        if meets_target(row["test"], baseline[row["seed"]], plan, require_final=False)
    ]
    selected.update({
        "selection_basis": "Validation metrics only. Test results do not enter the selection function.",
        "cases": cases, "references": references, "verification": verification,
        "run_artifact_hashes": anchors,
        "dataset_fingerprints": fingerprints,
        "paired_initial_hashes": initial_hashes,
        "replication": {
            "capacity": plan["replication_capacity"],
            "passing_seeds": replication_passes,
            "passing_count": len(replication_passes),
            "required": plan["required_exploration_successes"],
            "passed": len(replication_passes) >= plan["required_exploration_successes"],
            "criterion": "The original retention and acquisition comparison, without the new final-accuracy target",
        },
    })
    write_json(study / "selection.json", selected)
    print(json.dumps({
        "chosen_capacity": selected["chosen_capacity"],
        "validation_successes": selected["choices"],
        "replication": selected["replication"],
    }, indent=2))
    return selected


def confirmation_result(study, summary, plan, base):
    receipt = summary["confirmation"]
    if receipt is None:
        return {"status": "not_run", "reason": "No capacity met the validation target"}
    run = study / receipt["run"]
    if digest(run / "artifacts.json") != receipt["artifacts_sha256"]:
        raise AssertionError("Confirmation receipt changed")
    if read(run / "config.json") != condition_config(plan, base, summary["chosen_capacity"], confirmation=True):
        raise AssertionError("Confirmation differs from the predeclared procedure")
    checked = verify(run)
    child = read(run / "summary.json")
    baseline = {r["seed"]: r for r in child["results"] if r["method"] == "online"}
    pairs = []
    for row in child["results"]:
        if row["method"] == "replay":
            old = baseline[row["seed"]]
            pairs.append({
                "seed": row["seed"], "final_mean_accuracy": row["final_mean_accuracy"],
                "retention_improvement": old["old_task_retention_drop"] - row["old_task_retention_drop"],
                "acquisition_change": row["mean_acquisition_accuracy"] - old["mean_acquisition_accuracy"],
                "passes": meets_target(row, old, plan),
            })
    passed = sum(row["passes"] for row in pairs)
    return {
        "status": "completed", "passed": passed >= plan["required_confirmation_successes"],
        "passing_count": passed, "required": plan["required_confirmation_successes"],
        "pairs": pairs, "verification": checked,
        "results": [
            {"seed": row["seed"], "method": row["method"], **metrics_only(row),
             "training_seconds": row["training_seconds"]}
            for row in child["results"]
        ],
        "linear_references": child["linear_references"],
    }


def center_spread(values):
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def fmt(values, points=False):
    average, sd = center_spread(values)
    suffix = " pp" if points else "%"
    return f"{average * 100:.1f}{suffix} ± {sd * 100:.1f}"


def report(study, output):
    plan, base, _ = check_study_inputs(study)
    summary = read(study / "study_summary.json")
    selection = read(study / "selection.json")
    if summary["status"] != "completed":
        raise AssertionError("The study is incomplete")
    if digest(study / "selection.json") != summary["selection_sha256"]:
        raise AssertionError("Selection changed after confirmation")
    if digest(study / "manifest.json") != summary["manifest_sha256"]:
        raise AssertionError("The study manifest changed")
    for relative, expected in selection["run_artifact_hashes"].items():
        run = study / relative
        if digest(run / "artifacts.json") != expected:
            raise AssertionError("A selected condition receipt changed")
        for artifact, checksum in read(run / "artifacts.json").items():
            if digest(run / artifact) != checksum:
                raise AssertionError("An exploration artifact changed after verification")
    recomputed = select_capacity(selection["cases"], plan)
    if recomputed["chosen_capacity"] != selection["chosen_capacity"] or recomputed["choices"] != selection["choices"]:
        raise AssertionError("The saved selection does not follow the declared rule")
    confirmation = confirmation_result(study, summary, plan, base)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "confirmation.json", confirmation)
    checks = selection["verification"] + ([confirmation["verification"]] if confirmation["status"] == "completed" else [])
    verification = {
        "status": "passed",
        "checkpoints_evaluated": sum(row["checkpoints_evaluated"] for row in checks),
        "training_example_draws_checked": sum(row["training_example_draws_checked"] for row in checks),
        "data_and_initial_weights_paired_across_capacities": True,
        "selection_uses_validation_only": True,
        "confirmation_seeds_disjoint_from_pilot_and_exploration": True,
        "saved_inputs_and_outputs_unchanged": True,
    }
    write_json(output / "verification.json", verification)
    fields = [
        "capacity", "seed", "split", "final_mean_accuracy", "mean_acquisition_accuracy",
        "old_task_retention_drop", "training_seconds", "updates", "update_examples",
        "retained_examples",
    ]
    with (output / "exploration.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in selection["cases"]:
            for split in ("validation", "test"):
                writer.writerow({**row, **row[split], "split": split})
    if confirmation["status"] == "completed":
        with (output / "confirmation.csv").open("x", newline="") as f:
            fields = ["seed", "method", "final_mean_accuracy", "mean_acquisition_accuracy", "old_task_retention_drop", "training_seconds"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(confirmation["results"])
    make_plot(output, plan, selection, confirmation)
    cases = selection["cases"]
    choices = {row["capacity"]: row for row in selection["choices"]}
    chosen = selection["chosen_capacity"]
    replication = selection["replication"]
    lines = [
        "# Memory capacity experiment", "",
        f"Study: {study.name}", "",
        f"Completed in {summary['wall_seconds']:.1f} seconds. "
        f"Trained {summary['neural_models']} neural models. Evidence verification passed.", "",
        "## Selection and confirmation", "",
        (
            f"The smallest tested memory meeting the fixed validation criterion was {chosen} examples."
            if chosen is not None else
            "No tested memory met the fixed validation criterion. Confirmation was therefore skipped."
        ), "",
        f"Selection required at least {plan['final_accuracy_target'] * 100:g} percent final accuracy, "
        f"at least {plan['retention_margin'] * 100:g} percentage points less earlier accuracy lost "
        f"than current-only training, and at most {plan['acquisition_tolerance'] * 100:g} points lower "
        f"new-task acquisition accuracy, in {plan['required_exploration_successes']} of "
        f"{len(plan['exploration_seeds'])} exploration seeds. Selection used validation scores only.", "",
    ]
    if confirmation["status"] == "completed":
        lines.extend([
            f"On {len(plan['confirmation_seeds'])} fresh confirmation seeds, the selected memory met "
            f"the same target on held-out test data in {confirmation['passing_count']} seeds. "
            f"The required count was {confirmation['required']}. "
            f"Confirmation criterion {'passed' if confirmation['passed'] else 'failed'}.", "",
            "| Confirmation method | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost |",
            "| --- | ---: | ---: | ---: |",
        ])
        names = {"online": "Current examples", "replay": f"Replay with {chosen} examples", "full_history": "All earlier examples"}
        for method in ("online", "replay", "full_history"):
            rows = [row for row in confirmation["results"] if row["method"] == method]
            lines.append(
                f"| {names[method]} | {fmt([r['final_mean_accuracy'] for r in rows])} | "
                f"{fmt([r['mean_acquisition_accuracy'] for r in rows])} | "
                f"{fmt([r['old_task_retention_drop'] for r in rows], points=True)} |"
            )
        lines.extend(["", "Individual confirmation outcomes:", ""])
        for row in confirmation["pairs"]:
            lines.append(
                f"- Seed {row['seed']}: final accuracy {row['final_mean_accuracy']:.1%}; "
                f"retention improvement {row['retention_improvement'] * 100:.1f} points; "
                f"acquisition change {row['acquisition_change'] * 100:+.1f} points; "
                f"{'passed' if row['passes'] else 'failed'}."
            )
        lines.append("")
    lines.extend([
        "## Exploration results", "",
        "These test scores describe the fixed capacity sweep. They were not inputs to memory selection.", "",
        "| Memory examples | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost | Validation seeds meeting all targets |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for capacity in plan["capacities"]:
        rows = [row["test"] for row in cases if row["capacity"] == capacity]
        count = str(choices[capacity]["passing_count"]) if capacity else "reference"
        lines.append(
            f"| {capacity} | {fmt([r['final_mean_accuracy'] for r in rows])} | "
            f"{fmt([r['mean_acquisition_accuracy'] for r in rows])} | "
            f"{fmt([r['old_task_retention_drop'] for r in rows], points=True)} | {count} |"
        )
    lines.extend([
        "",
        "Values show means and sample standard deviations across seeds. "
        "Earlier accuracy lost is a signed difference; negative values mean improvement.",
        "",
        f"The original {replication['capacity']}-example comparison met its retention and acquisition "
        f"criterion in {replication['passing_count']} of {len(plan['exploration_seeds'])} fresh seeds. "
        f"The predeclared required count was {replication['required']}. "
        f"Replication criterion {'passed' if replication['passed'] else 'failed'}. "
        "This criterion does not include the new final-accuracy target.",
        "",
        f"The full-history reference reached {fmt([r['final_mean_accuracy'] for r in selection['references']['full_history']])}. "
        f"The separate linear reference reached {fmt([r['final_mean_accuracy'] for r in selection['references']['linear']])}.",
        "",
        "## Cost, verification, and limits", "",
        f"Every neural model used {base['tasks'] * base['steps_per_task']:,} updates and "
        f"{base['tasks'] * base['steps_per_task'] * base['batch_size']:,} example presentations. "
        "Architecture, task distribution, optimizer, and replay fraction were held fixed.",
        "",
        f"Verification reloaded {verification['checkpoints_evaluated']} checkpoints and audited "
        f"{verification['training_example_draws_checked']:,} training-example draws. "
        "Exploration datasets, task orders, and starting weights were identical across capacities within each seed.",
        "",
        "The complete study time includes condition runs, process startup, and validation selection. "
        "The final confirmation readback and report generation occur afterward. Per-model training "
        "times are in the CSV files. Similar timings do not establish identical energy use.",
        "",
        "Memory counts describe retained example access. They do not include neural weights, optimizer "
        "state, the currently available training block, or the benchmark's evaluation data.",
        "",
        "The selected size, if any, is only the smallest tested size meeting this particular target. "
        "It is not a universal minimum. A failure to meet a threshold is retained as a negative result.",
        "",
        "These remain supervised synthetic tasks with explicit machine identities and linear rule "
        "boundaries. The simple linear reference uses a different learning procedure and 216 fitted "
        "parameters. The experiment does not establish language ability, autonomous exploration, "
        "general concept learning, or the elimination of large-model pre-training.",
        "",
        "## Files", "",
        "- results.png and results.svg contain the charts.",
        "- exploration.csv contains both validation and test scores for every capacity and seed.",
        "- confirmation.json contains the fresh-seed outcome, including failures or the reason it was skipped.",
        "- verification.json records the evidence checks.",
        "- The parent directory retains the frozen plan, source, input configurations, selection, and all condition runs.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))
    shutil.copy2(Path(__file__), output / "analysis_source.py")
    write_json(output / "provenance.json", {
        "study_summary_sha256": digest(study / "study_summary.json"),
        "selection_sha256": digest(study / "selection.json"),
        "analysis_source_sha256": digest(output / "analysis_source.py"),
    })
    print("Report: " + str(output / "REPORT.md"))


def make_plot(output, plan, selection, confirmation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "#fafaf8",
        "axes.facecolor": "#fafaf8", "savefig.facecolor": "#fafaf8",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    capacities = plan["capacities"]
    settings = [
        ("final_mean_accuracy", "Final accuracy on unfamiliar examples"),
        ("old_task_retention_drop", "Earlier learning lost by the end"),
        ("mean_acquisition_accuracy", "Accuracy when a new rule has just been learned"),
    ]
    for ax, (key, title) in zip(axes.flat, settings):
        averages, spreads = [], []
        for capacity in capacities:
            values = [row["test"][key] for row in selection["cases"] if row["capacity"] == capacity]
            average, spread = center_spread(values)
            averages.append(average)
            spreads.append(spread)
        ax.errorbar(capacities, averages, yerr=spreads, color="#26836f", marker="o", capsize=4)
        reference = mean(row[key] for row in selection["references"]["full_history"])
        ax.axhline(reference, color="#3274a1", linestyle="--", label="All earlier examples")
        if key == "final_mean_accuracy":
            linear = mean(row[key] for row in selection["references"]["linear"])
            ax.axhline(linear, color="#777777", linestyle=":", label="Separate linear reference")
            ax.axhline(plan["final_accuracy_target"], color="#b68c20", linestyle=":", label="Working target")
            ax.legend(fontsize=8, loc="lower right")
        if selection["chosen_capacity"] is not None:
            ax.axvline(selection["chosen_capacity"], color="#777777", linewidth=1, alpha=0.5)
        ax.set_xscale("symlog", linthresh=max(capacities[1], 1), base=2)
        ax.set_xticks(capacities, [str(v) for v in capacities], fontsize=8)
        ax.set_xlabel("Retained replay examples")
        ax.set_title(title)
        ax.set_ylim(-0.05 if key == "old_task_retention_drop" else 0, 1)
    ax = axes[1, 1]
    if confirmation["status"] == "completed":
        methods = ("online", "replay", "full_history")
        labels = ("Current examples", "Selected memory", "All earlier examples")
        for index, method in enumerate(methods):
            values = [row["final_mean_accuracy"] for row in confirmation["results"] if row["method"] == method]
            average, spread = center_spread(values)
            ax.bar(index, average, color=("#c46a35", "#26836f", "#3274a1")[index], width=0.6)
            ax.errorbar(index, average, yerr=spread, color="#202020", capsize=4, fmt="none")
            ax.scatter(index + np.linspace(-0.15, 0.15, len(values)), values, color="#202020", s=14, zorder=3)
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.axhline(plan["final_accuracy_target"], color="#b68c20", linestyle=":")
        ax.set_title(f"Confirmation on {len(plan['confirmation_seeds'])} new seeds")
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No tested capacity met every validation target.\nConfirmation was skipped.", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Confirmation")
        ax.set_ylim(0, 1)
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("How much replay memory is needed?", fontsize=17, x=0.07, ha="left")
    fig.text(
        0.07, 0.015,
        f"{len(plan['exploration_seeds'])} exploration seeds. Error bars show sample standard deviation. "
        "The vertical line marks the size selected using validation scores.\n"
        "The neural update budget is fixed. The linear reference uses a different learning procedure.",
        fontsize=8, color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.94), h_pad=3)
    fig.savefig(output / "results.png", dpi=150)
    fig.savefig(output / "results.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "report"])
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
