"""Create a separate report from a verified, completed run."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from continual_lab import digest, write_json
from verify_run import verify


LABELS = {
    "full_history": "All earlier examples",
    "online": "Current examples",
    "replay": "Small replay memory",
}
COLORS = {"full_history": "#3274a1", "online": "#c46a35", "replay": "#26836f"}


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def percent(values):
    mean, sd = mean_sd(values)
    return f"{mean * 100:.1f}% ± {sd * 100:.1f}"


def points(values):
    mean, sd = mean_sd(values)
    return f"{mean * 100:.1f} pp ± {sd * 100:.1f}"


def write_csv(path, rows, columns):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate(run_dir, output):
    verification = verify(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "verification.json", verification)
    c = json.loads((run_dir / "config.json").read_text())
    s = json.loads((run_dir / "summary.json").read_text())
    methods = c["methods"]
    rows = s["results"]
    references = s["linear_references"]
    grouped = {m: [r for r in rows if r["method"] == m] for m in methods}
    write_csv(output / "results.csv", rows, [
        "seed", "method", "final_mean_accuracy", "mean_acquisition_accuracy",
        "old_task_retention_drop", "updates", "update_examples",
        "training_seconds", "evaluation_seconds", "memory_maintenance_seconds",
        "method_wall_seconds", "parameter_count", "retained_examples",
        "retained_example_payload_bytes",
    ])
    write_csv(output / "learning_curves.csv", [x for r in rows for x in r["curves"]], [
        "seed", "method", "stage", "task", "step", "validation_accuracy",
        "unique_current_examples_used", "current_examples_available",
        "total_update_examples", "stage_training_seconds",
    ])
    write_csv(output / "stages.csv", [x for r in rows for x in r["stages"]], [
        "seed", "method", "stage", "task", "current_test_accuracy",
        "mean_seen_test_accuracy", "unique_current_examples_used",
        "past_memory_examples_during_stage", "retained_examples_after_stage",
        "stage_training_seconds",
    ])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "#fafaf8",
        "axes.facecolor": "#fafaf8", "savefig.facecolor": "#fafaf8",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    final_groups = [grouped[m] for m in methods] + [references]
    labels = [LABELS[m] for m in methods] + ["Separate linear\nreference"]
    colors = [COLORS[m] for m in methods] + ["#777777"]
    for index, group in enumerate(final_groups):
        values = [r["final_mean_accuracy"] for r in group]
        mean, sd = mean_sd(values)
        axes[0, 0].bar(index, mean, color=colors[index], alpha=0.85, width=0.6)
        axes[0, 0].errorbar(index, mean, yerr=sd, color="#202020", capsize=4, fmt="none")
        offsets = np.linspace(-0.15, 0.15, len(values))
        axes[0, 0].scatter(index + offsets, values, color="#202020", s=12, zorder=5)
    axes[0, 0].set_xticks(range(len(labels)), labels, fontsize=8)
    axes[0, 0].set_title("Final accuracy on unfamiliar examples")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axhline(0.25, color="#aaaaaa", linestyle=":", linewidth=1)
    for method in methods:
        stage_values = np.array([
            [stage["mean_seen_test_accuracy"] for stage in r["stages"]]
            for r in grouped[method]
        ])
        mean = stage_values.mean(axis=0)
        sd = stage_values.std(axis=0, ddof=1) if len(stage_values) > 1 else np.zeros_like(mean)
        x = np.arange(1, c["tasks"] + 1)
        axes[0, 1].plot(x, mean, color=COLORS[method], marker="o", label=LABELS[method])
        axes[0, 1].fill_between(x, mean - sd, mean + sd, color=COLORS[method], alpha=0.1)
    axes[0, 1].set_title("Accuracy on machines encountered so far")
    axes[0, 1].set_xlabel("Number of machines introduced")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(fontsize=8, loc="lower left")
    for index, method in enumerate(methods):
        values = [r["old_task_retention_drop"] for r in grouped[method]]
        mean, sd = mean_sd(values)
        axes[1, 0].bar(index, mean, color=COLORS[method], alpha=0.85, width=0.6)
        axes[1, 0].errorbar(index, mean, yerr=sd, color="#202020", capsize=4, fmt="none")
        axes[1, 0].scatter(index + np.linspace(-0.15, 0.15, len(values)), values, color="#202020", s=12, zorder=5)
    axes[1, 0].set_xticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=8)
    axes[1, 0].axhline(0, color="#aaaaaa", linewidth=1)
    axes[1, 0].set_title("Earlier learning lost by the end")
    axes[1, 0].set_ylabel("Accuracy at acquisition minus final accuracy")
    for method in methods:
        all_curves = [
            [next(row["validation_accuracy"] for row in r["curves"] if row["stage"] == stage and row["step"] == step)
             for step in c["curve_steps"]]
            for r in grouped[method] for stage in range(c["tasks"])
        ]
        values = np.asarray(all_curves)
        axes[1, 1].plot(c["curve_steps"], values.mean(axis=0), color=COLORS[method], marker=".", label=LABELS[method])
    axes[1, 1].axhline(c["target_accuracy"], color="#aaaaaa", linestyle=":", linewidth=1)
    axes[1, 1].set_title("Learning the current machine")
    axes[1, 1].set_xlabel("Training updates within a machine block")
    axes[1, 1].set_ylim(0, 1)
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle("Learning new rules and retaining earlier rules", fontsize=17, x=0.07, ha="left")
    fig.text(
        0.07, 0.015,
        f"{len(c['seeds'])} seeds. Bars and bands show sample standard deviation. "
        "The linear reference uses separate models and a different training procedure.\n"
        "All neural methods use the same architecture, initial weights per seed, update count, and batch size.",
        fontsize=8, color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.99, 0.94), h_pad=3)
    fig.savefig(output / "results.png", dpi=150)
    fig.savefig(output / "results.svg")
    plt.close(fig)
    lines = [
        "# First learning and retention experiment", "",
        f"Run: {s['run_id']}", "",
        f"Status: completed. Evidence verification: {verification['status']}.", "",
        "## Result", "",
        f"The predeclared replay criterion was met in {s['decision'].get('passing_seeds', 0)} "
        f"of {len(c['seeds'])} seeds. Verdict: {s['decision']['verdict']}.", "",
        f"The criterion requires at least a {c['decision']['retention_margin'] * 100:g} percentage point "
        f"reduction in lost earlier accuracy, with no more than a "
        f"{c['decision']['acquisition_tolerance'] * 100:g} percentage point reduction in new-task "
        f"acquisition accuracy, in at least {c['decision']['required_seed_wins']} of "
        f"{len(c['seeds'])} seeds.", "",
        "| Method | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost | Training seconds | Retained examples |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        group = grouped[method]
        lines.append(
            f"| {LABELS[method]} | {percent([r['final_mean_accuracy'] for r in group])} | "
            f"{percent([r['mean_acquisition_accuracy'] for r in group])} | "
            f"{points([r['old_task_retention_drop'] for r in group])} | "
            f"{statistics.mean(r['training_seconds'] for r in group):.2f} | {group[0]['retained_examples']} |"
        )
    lines.extend([
        "",
        "Accuracy values show the mean and sample standard deviation across seeds. "
        "Earlier accuracy lost is a signed percentage point difference. A negative value means improvement.",
        "",
        f"The separate linear reference reached {percent([r['final_mean_accuracy'] for r in references])} "
        f"using {references[0]['parameter_count']} fitted numbers. Each neural model has "
        f"{rows[0]['parameter_count']:,} parameters. This reference is useful because the "
        "simulated rules have linear boundaries and the machine identity is given. "
        "It is not a budget-matched neural method.",
        "",
        "## Cost and interpretation", "",
        f"Each neural method used {rows[0]['updates']:,} updates and "
        f"{rows[0]['update_examples']:,} sampled example presentations per seed. "
        f"The full experiment took {s['wall_seconds']:.1f} seconds. "
        f"Peak process memory was {s['peak_process_memory_mib']:.1f} MiB.",
        "",
        "Training time includes batch sampling and access checks. Memory maintenance, evaluation, "
        "and method wall time are separate fields in results.csv. The full run time also includes "
        "data creation, model setup, and evidence saving. Report generation and verification happen afterward.",
        "",
        "Retained-example counts describe what each learning method may access. "
        "The benchmark process holds evaluation data and the complete simulated world for all methods. "
        "Its process memory is not a measurement of each method's deployment memory.",
        "",
        "All methods can access the same new block of training examples. Their allocation of the "
        "fixed update budget differs. Replay spends some updates on old examples. "
        "The all-history method samples uniformly from all available tasks. "
        "It is a reference with more memory, not a tuned upper bound.",
        "",
        "This experiment tests supervised learning in one synthetic task family. "
        "It does not establish language ability, autonomous exploration, brain-like learning, "
        "transfer to unrelated task families, or a replacement for large-model pre-training. "
        "The labels are supplied by the simulator, and machine identities are explicit.",
        "",
        "## Evidence", "",
        "- results.png and results.svg contain the plots.",
        "- results.csv contains every seed and method result.",
        "- learning_curves.csv records validation accuracy and example use.",
        "- stages.csv records acquisition and retention after each task.",
        "- verification.json records the independent readback checks.",
        "- The parent run directory contains frozen source, data, checkpoints, batch IDs, and event logs.",
        "",
        "The first recorded step above the target accuracy is an exploratory learning-speed measure. "
        "It is not a minimum required sample count: all examples in the current block were available, "
        "checks occur at discrete steps, and a threshold crossing need not persist.",
        "",
        "Future experiments should use new run directories and retain this evidence unchanged.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))
    shutil.copy2(Path(__file__), output / "report_source.py")
    write_json(output / "provenance.json", {
        "input_summary_sha256": digest(run_dir / "summary.json"),
        "input_artifacts_sha256": digest(run_dir / "artifacts.json"),
        "report_source_sha256": digest(output / "report_source.py"),
        "matplotlib_version": matplotlib.__version__,
    })
    print("Report: " + str(output / "REPORT.md"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run.resolve()
    generate(run_dir, args.output.resolve() if args.output else run_dir / "report")


if __name__ == "__main__":
    main()
