"""Compare the share of training devoted to replay, at fixed memory and cost."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from continual_lab import digest, validate_config, write_json
from memory_sweep import meets_target


ROOT = Path(__file__).resolve().parent


def validate_plan(plan, base):
    required = {
        "experiment", "base_config", "memory_capacity", "replay_fractions",
        "reference_fraction", "exploration_seeds", "confirmation_seeds",
        "excluded_seeds", "final_accuracy_target", "retention_margin",
        "acquisition_tolerance", "required_exploration_successes",
        "required_confirmation_successes", "wall_seconds_limit",
    }
    if set(plan) != required:
        raise ValueError(f"Plan keys differ: {set(plan) ^ required}")
    values = plan["replay_fractions"]
    if (
        len(values) < 2 or values != sorted(set(values)) or values[0] != 0
        or any(type(v) not in (int, float) or not 0 <= v < 1 for v in values)
    ):
        raise ValueError("Replay fractions must be ordered and unique, starting at zero")
    if any(not float(v * base["batch_size"]).is_integer() for v in values):
        raise ValueError("Each replay fraction must specify a whole number of batch examples")
    if plan["reference_fraction"] not in values[1:]:
        raise ValueError("The reference must be a positive tested replay fraction")
    if type(plan["memory_capacity"]) is not int or plan["memory_capacity"] < 1:
        raise ValueError("Memory capacity must be positive")
    for phase in ("exploration", "confirmation"):
        seeds = plan[phase + "_seeds"]
        needed = plan["required_" + phase + "_successes"]
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("Seed lists must contain unique nonnegative integers")
        if type(needed) is not int or not 1 <= needed <= len(seeds):
            raise ValueError("Required successes must fit the seed count")
    exploration = set(plan["exploration_seeds"])
    confirmation = set(plan["confirmation_seeds"])
    excluded = set(base["seeds"]) | set(plan["excluded_seeds"])
    if exploration & confirmation or (exploration | confirmation) & excluded:
        raise ValueError("Exploration and confirmation must use disjoint, unobserved seeds")
    if not 0 < plan["final_accuracy_target"] <= 1:
        raise ValueError("Invalid final accuracy target")
    if not 0 <= plan["retention_margin"] <= 1 or not 0 <= plan["acquisition_tolerance"] <= 1:
        raise ValueError("Invalid comparison margin")
    if type(plan["wall_seconds_limit"]) is not int or plan["wall_seconds_limit"] < 1:
        raise ValueError("The study time limit must be positive")
    validate_config(base)


def condition_config(plan, base, fraction, phase="exploration"):
    if phase not in ("exploration", "confirmation", "reference"):
        raise ValueError("Unknown phase")
    c = copy.deepcopy(base)
    seed_phase = "exploration" if phase == "exploration" else "confirmation"
    c.update(
        experiment=f"{plan['experiment']}_{phase}_{fraction:g}",
        seeds=list(plan[seed_phase + "_seeds"]),
        memory_capacity=plan["memory_capacity"],
        replay_fraction=fraction,
    )
    c["methods"] = (
        ["full_history", "online", "replay"] if phase == "confirmation" else
        ["replay"] if phase == "reference" or fraction > 0 else
        ["full_history", "online"]
    )
    c["decision"] = {
        "retention_margin": plan["retention_margin"],
        "acquisition_tolerance": plan["acquisition_tolerance"],
        "required_seed_wins": plan["required_" + seed_phase + "_successes"],
    }
    validate_config(c)
    return c


def select_fraction(cases, plan):
    baseline_rows = [row for row in cases if row["fraction"] == 0]
    seeds = set(plan["exploration_seeds"])
    if len(baseline_rows) != len(seeds) or {r["seed"] for r in baseline_rows} != seeds:
        raise ValueError("Missing or duplicate current-only baselines")
    baseline = {row["seed"]: row["validation"] for row in baseline_rows}
    choices = []
    for fraction in plan["replay_fractions"][1:]:
        rows = [row for row in cases if row["fraction"] == fraction]
        if len(rows) != len(seeds) or {r["seed"] for r in rows} != seeds:
            raise ValueError("Missing or duplicate replay cases")
        passed = [
            row["seed"] for row in rows
            if meets_target(row["validation"], baseline[row["seed"]], plan)
        ]
        choices.append({
            "fraction": fraction, "passing_seeds": passed, "passing_count": len(passed),
            "qualifies": len(passed) >= plan["required_exploration_successes"],
        })
    qualifying = [row["fraction"] for row in choices if row["qualifies"]]
    return {"chosen_fraction": min(qualifying) if qualifying else None, "choices": choices}


def run_study(plan_path, output_root):
    plan = json.loads(plan_path.read_text())
    base = json.loads((ROOT / plan["base_config"]).read_text())
    validate_plan(plan, base)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = output_root / (stamp + "-allocation-" + uuid.uuid4().hex[:8])
    study.mkdir(parents=True, exist_ok=False)
    (study / "configs").mkdir()
    (study / "logs").mkdir()
    started = time.perf_counter()
    deadline = started + plan["wall_seconds_limit"]

    def emit(event, details):
        with (study / "events.jsonl").open("a") as f:
            f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}) + "\n")
        print(event + ": " + json.dumps(details), flush=True)

    emit("study_started", {"path": str(study)})
    write_json(study / "plan.json", plan)
    write_json(study / "base_config.json", base)
    source_hashes = {}
    for path in sorted([
        *ROOT.glob("*.py"), *ROOT.glob("*.md"), *ROOT.glob("configs/*.json"),
        *ROOT.glob("tests/*.py"), ROOT / "requirements.txt",
    ]):
        relative = path.relative_to(ROOT)
        destination = study / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = digest(destination)
    config_hashes = {}
    for fraction in plan["replay_fractions"]:
        path = study / "configs" / f"fraction_{fraction:g}.json"
        write_json(path, condition_config(plan, base, fraction))
        config_hashes[str(path.relative_to(study))] = digest(path)
    write_json(study / "manifest.json", {
        "plan_sha256": digest(study / "plan.json"),
        "base_sha256": digest(study / "base_config.json"),
        "source_sha256": source_hashes, "config_sha256": config_hashes,
    })
    emit("plan_frozen", {"manifest_sha256": digest(study / "manifest.json")})

    def invoke(command, log_name):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("The study time limit was reached")
        log = study / "logs" / log_name
        with log.open("x") as f:
            result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=remaining)
        if result.returncode:
            raise RuntimeError(f"Command failed with exit {result.returncode}; retained log: {log}")

    def condition(path, name):
        target = study / name
        target.mkdir(parents=True, exist_ok=False)
        invoke(
            [sys.executable, str(ROOT / "run_experiment.py"), "--config", str(path), "--output", str(target)],
            name.replace("/", "_") + ".log",
        )
        runs = [p for p in target.iterdir() if p.is_dir()]
        if len(runs) != 1 or not (runs[0] / "summary.json").exists():
            raise RuntimeError("Expected one completed run")
        state = json.loads((runs[0] / "summary.json").read_text())
        if state["status"] != "completed":
            raise RuntimeError("A condition did not complete")
        return {
            "run": str(runs[0].relative_to(study)), "artifacts_sha256": digest(runs[0] / "artifacts.json"),
            "wall_seconds": state["wall_seconds"], "model_count": len(state["results"]),
        }

    completed = []
    try:
        for fraction in plan["replay_fractions"]:
            emit("condition_started", {"fraction": fraction})
            receipt = condition(study / "configs" / f"fraction_{fraction:g}.json", f"exploration/fraction_{fraction:g}")
            receipt["fraction"] = fraction
            completed.append(receipt)
            emit("condition_completed", receipt)
        write_json(study / "exploration.json", completed)
        emit("selection_started", {"basis": "validation scores only"})
        invoke(
            [sys.executable, str(ROOT / "analyze_replay_allocation.py"), "select", str(study)],
            "selection.log",
        )
        selected = json.loads((study / "selection.json").read_text())
        chosen = selected["chosen_fraction"]
        emit("fraction_selected", {"fraction": chosen, "selection_sha256": digest(study / "selection.json")})
        confirmation = reference = None
        if chosen is not None:
            path = study / "configs" / "confirmation.json"
            write_json(path, condition_config(plan, base, chosen, "confirmation"))
            emit("confirmation_started", {"fraction": chosen, "config_sha256": digest(path)})
            confirmation = condition(path, "confirmation")
            emit("confirmation_completed", confirmation)
            if chosen != plan["reference_fraction"]:
                path = study / "configs" / "reference.json"
                write_json(path, condition_config(plan, base, plan["reference_fraction"], "reference"))
                emit("paired_reference_started", {"fraction": plan["reference_fraction"], "config_sha256": digest(path)})
                reference = condition(path, "reference")
                emit("paired_reference_completed", reference)
        else:
            emit("confirmation_skipped", {"reason": "No fraction met the fixed validation criterion"})
        summary = {
            "status": "completed", "study_id": study.name, "chosen_fraction": chosen,
            "exploration": completed, "confirmation": confirmation, "paired_reference": reference,
            "selection_sha256": digest(study / "selection.json"),
            "manifest_sha256": digest(study / "manifest.json"),
            "wall_seconds": time.perf_counter() - started,
            "neural_models": sum(r["model_count"] for r in completed) + sum(
                r["model_count"] for r in (confirmation, reference) if r is not None
            ),
        }
        write_json(study / "study_summary.json", summary)
        emit("study_completed", {"wall_seconds": summary["wall_seconds"], "neural_models": summary["neural_models"]})
        print("Saved study: " + str(study), flush=True)
        return study
    except BaseException as exc:
        incomplete = isinstance(exc, (TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt))
        failure = {
            "status": "incomplete" if incomplete else "failed", "error_type": type(exc).__name__,
            "error": str(exc), "completed_conditions": completed, "traceback": traceback.format_exc(),
        }
        write_json(study / "failure.json", failure)
        emit("study_stopped", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/replay_allocation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "studies")
    args = parser.parse_args()
    run_study(args.plan.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
