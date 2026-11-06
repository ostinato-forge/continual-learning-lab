"""Run a fixed memory-capacity study with fresh confirmation seeds."""

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


ROOT = Path(__file__).resolve().parent


def validate_plan(plan, base):
    required = {
        "experiment", "base_config", "capacities", "exploration_seeds",
        "confirmation_seeds", "replication_capacity", "final_accuracy_target",
        "retention_margin", "acquisition_tolerance",
        "required_exploration_successes", "required_confirmation_successes",
        "wall_seconds_limit",
    }
    if set(plan) != required:
        raise ValueError(f"Plan keys differ: {set(plan) ^ required}")
    capacities = plan["capacities"]
    if (
        len(capacities) < 2 or capacities != sorted(set(capacities))
        or capacities[0] != 0
        or any(type(value) is not int or value < 0 for value in capacities)
    ):
        raise ValueError("Capacities must be ordered, unique integers starting with zero")
    if plan["replication_capacity"] not in capacities[1:]:
        raise ValueError("Replication capacity must be a positive tested capacity")
    for phase in ("exploration", "confirmation"):
        seeds = plan[phase + "_seeds"]
        needed = plan["required_" + phase + "_successes"]
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("Seed lists must contain unique nonnegative integers")
        if type(needed) is not int or not 1 <= needed <= len(seeds):
            raise ValueError("Required successes must fit the seed list")
    exploration = set(plan["exploration_seeds"])
    confirmation = set(plan["confirmation_seeds"])
    if exploration & confirmation or (exploration | confirmation) & set(base["seeds"]):
        raise ValueError("Pilot, exploration, and confirmation seeds must be disjoint")
    if not 0 < plan["final_accuracy_target"] <= 1:
        raise ValueError("Invalid accuracy target")
    if not 0 <= plan["retention_margin"] <= 1 or not 0 <= plan["acquisition_tolerance"] <= 1:
        raise ValueError("Invalid comparison margin")
    if type(plan["wall_seconds_limit"]) is not int or plan["wall_seconds_limit"] < 1:
        raise ValueError("The study needs a positive time limit")
    validate_config(base)


def condition_config(plan, base, capacity, confirmation=False):
    c = copy.deepcopy(base)
    phase = "confirmation" if confirmation else "exploration"
    c["experiment"] = f"{plan['experiment']}_{phase}_{capacity}"
    c["seeds"] = list(plan[phase + "_seeds"])
    c["memory_capacity"] = capacity
    c["methods"] = (
        ["full_history", "online", "replay"] if confirmation else
        ["full_history", "online"] if capacity == 0 else ["replay"]
    )
    c["decision"] = {
        "retention_margin": plan["retention_margin"],
        "acquisition_tolerance": plan["acquisition_tolerance"],
        "required_seed_wins": plan["required_" + phase + "_successes"],
    }
    validate_config(c)
    return c


def meets_target(candidate, baseline, plan, require_final=True):
    retained = baseline["old_task_retention_drop"] - candidate["old_task_retention_drop"]
    acquisition = candidate["mean_acquisition_accuracy"] - baseline["mean_acquisition_accuracy"]
    return bool(
        retained >= plan["retention_margin"]
        and acquisition >= -plan["acquisition_tolerance"]
        and (not require_final or candidate["final_mean_accuracy"] >= plan["final_accuracy_target"])
    )


def select_capacity(cases, plan):
    """Choose using validation metrics only. Test metrics are ignored."""
    baseline = {row["seed"]: row["validation"] for row in cases if row["capacity"] == 0}
    choices = []
    for capacity in plan["capacities"][1:]:
        group = [row for row in cases if row["capacity"] == capacity]
        if {row["seed"] for row in group} != set(plan["exploration_seeds"]) or len(group) != len(baseline):
            raise ValueError("Missing or duplicate exploration cases")
        passed = [
            row["seed"] for row in group
            if meets_target(row["validation"], baseline[row["seed"]], plan)
        ]
        choices.append({
            "capacity": capacity, "passing_seeds": passed, "passing_count": len(passed),
            "qualifies": len(passed) >= plan["required_exploration_successes"],
        })
    qualifying = [row["capacity"] for row in choices if row["qualifies"]]
    return {"chosen_capacity": min(qualifying) if qualifying else None, "choices": choices}


def run_study(plan_path, output_root):
    plan = json.loads(plan_path.read_text())
    base = json.loads((ROOT / plan["base_config"]).read_text())
    validate_plan(plan, base)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = output_root / (stamp + "-" + uuid.uuid4().hex[:8])
    study.mkdir(parents=True, exist_ok=False)
    (study / "configs").mkdir()
    (study / "logs").mkdir()
    started = time.perf_counter()
    deadline = started + plan["wall_seconds_limit"]

    def emit(event, details):
        entry = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        with (study / "events.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(event + ": " + json.dumps(details), flush=True)

    emit("study_started", {"path": str(study)})
    write_json(study / "plan.json", plan)
    write_json(study / "base_config.json", base)
    files = [
        *ROOT.glob("*.py"), *ROOT.glob("*.md"), *ROOT.glob("configs/*.json"),
        *ROOT.glob("tests/*.py"), ROOT / "requirements.txt",
    ]
    source_hashes = {}
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        destination = study / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = digest(destination)
    config_hashes = {}
    for capacity in plan["capacities"]:
        path = study / "configs" / f"capacity_{capacity}.json"
        write_json(path, condition_config(plan, base, capacity))
        config_hashes[str(path.relative_to(study))] = digest(path)
    write_json(study / "manifest.json", {
        "plan_sha256": digest(study / "plan.json"),
        "base_sha256": digest(study / "base_config.json"),
        "source_sha256": source_hashes, "config_sha256": config_hashes,
    })
    emit("plan_frozen", {"manifest_sha256": digest(study / "manifest.json")})

    def invoke(command, log_path):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("The study time limit was reached")
        with log_path.open("x") as f:
            result = subprocess.run(
                command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=remaining,
            )
        if result.returncode:
            raise RuntimeError(f"Command failed with exit {result.returncode}; retained log: {log_path}")

    def run_condition(config, name):
        target = study / name
        target.mkdir(parents=True, exist_ok=False)
        invoke(
            [sys.executable, str(ROOT / "run_experiment.py"), "--config", str(config), "--output", str(target)],
            study / "logs" / (name.replace("/", "_") + ".log"),
        )
        children = [p for p in target.iterdir() if p.is_dir()]
        if len(children) != 1 or not (children[0] / "summary.json").exists():
            raise RuntimeError("The condition did not produce exactly one completed run")
        run_dir = children[0]
        summary = json.loads((run_dir / "summary.json").read_text())
        if summary["status"] != "completed":
            raise RuntimeError("The condition is incomplete")
        return {
            "run": str(run_dir.relative_to(study)),
            "artifacts_sha256": digest(run_dir / "artifacts.json"),
            "wall_seconds": summary["wall_seconds"],
            "model_count": len(summary["results"]),
        }

    completed = []
    try:
        for capacity in plan["capacities"]:
            emit("condition_started", {"capacity": capacity})
            receipt = run_condition(
                study / "configs" / f"capacity_{capacity}.json",
                f"exploration/capacity_{capacity}",
            )
            receipt["capacity"] = capacity
            completed.append(receipt)
            emit("condition_completed", receipt)
        write_json(study / "exploration.json", completed)
        emit("selection_started", {"basis": "validation scores only"})
        invoke(
            [sys.executable, str(ROOT / "analyze_memory_sweep.py"), "select", str(study)],
            study / "logs" / "selection.log",
        )
        selection = json.loads((study / "selection.json").read_text())
        chosen = selection["chosen_capacity"]
        emit("candidate_selected", {"capacity": chosen, "selection_sha256": digest(study / "selection.json")})
        confirmation = None
        if chosen is not None:
            config = study / "configs" / "confirmation.json"
            write_json(config, condition_config(plan, base, chosen, confirmation=True))
            emit("confirmation_started", {"capacity": chosen, "config_sha256": digest(config)})
            confirmation = run_condition(config, "confirmation")
            emit("confirmation_completed", confirmation)
        else:
            emit("confirmation_skipped", {"reason": "No tested capacity met the fixed validation criterion"})
        summary = {
            "status": "completed", "study_id": study.name,
            "chosen_capacity": chosen, "exploration": completed,
            "confirmation": confirmation,
            "selection_sha256": digest(study / "selection.json"),
            "manifest_sha256": digest(study / "manifest.json"),
            "wall_seconds": time.perf_counter() - started,
            "neural_models": sum(row["model_count"] for row in completed) + (
                confirmation["model_count"] if confirmation else 0
            ),
        }
        write_json(study / "study_summary.json", summary)
        emit("study_completed", {"wall_seconds": summary["wall_seconds"], "neural_models": summary["neural_models"]})
        print("Saved study: " + str(study), flush=True)
        return study
    except BaseException as exc:
        incomplete = isinstance(exc, (TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt))
        failure = {
            "status": "incomplete" if incomplete else "failed",
            "error_type": type(exc).__name__, "error": str(exc),
            "completed_conditions": completed, "traceback": traceback.format_exc(),
        }
        write_json(study / "failure.json", failure)
        emit("study_stopped", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/memory_sweep.json")
    parser.add_argument("--output", type=Path, default=ROOT / "studies")
    args = parser.parse_args()
    run_study(args.plan.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
