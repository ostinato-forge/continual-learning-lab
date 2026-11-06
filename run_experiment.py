"""Run a frozen experiment and preserve its inputs, traces, and outcomes."""

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

from continual_lab import (
    digest, fit_linear_reference, fit_method, make_world,
    save_world, validate_config, write_json,
)


ROOT = Path(__file__).resolve().parent


def source_paths():
    return sorted([
        *ROOT.glob("*.py"), *ROOT.glob("*.md"),
        *ROOT.glob("tests/*.py"), *ROOT.glob("configs/*.json"),
        ROOT / "requirements.txt",
    ])


def decision(results, c):
    paired = []
    for seed in c["seeds"]:
        arms = {r["method"]: r for r in results if r["seed"] == seed}
        if not {"online", "replay"} <= set(arms):
            return {"verdict": "not_evaluated", "reason": "Online and replay methods are both required"}
        online, replay = arms["online"], arms["replay"]
        improvement = online["old_task_retention_drop"] - replay["old_task_retention_drop"]
        acquisition_change = replay["mean_acquisition_accuracy"] - online["mean_acquisition_accuracy"]
        passed = (
            improvement >= c["decision"]["retention_margin"]
            and acquisition_change >= -c["decision"]["acquisition_tolerance"]
        )
        paired.append({
            "seed": seed, "retention_improvement": improvement,
            "acquisition_change": acquisition_change, "passes": passed,
        })
    wins = sum(p["passes"] for p in paired)
    return {
        "verdict": "supported_in_this_pilot" if wins >= c["decision"]["required_seed_wins"] else "not_supported_in_this_pilot",
        "paired_results": paired, "passing_seeds": wins,
        "required_seeds": c["decision"]["required_seed_wins"],
        "scope": "Replay retention at equal update counts in this synthetic task family",
    }


def run(config_path, output_root):
    c = json.loads(config_path.read_text())
    validate_config(c)
    torch.set_num_threads(c["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = stamp + "-" + uuid.uuid4().hex[:8]
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + c["wall_seconds_limit"]
    events = directory / "events.jsonl"

    def emit(event, details):
        entry = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        with events.open("a") as f:
            f.write(json.dumps(entry, allow_nan=False) + "\n")
        if event == "stage_completed":
            print(
                f"seed {details['seed']} | {details['method']} | "
                f"machine {details['task']} | current {details['current_test_accuracy']:.1%} | "
                f"seen {details['mean_seen_test_accuracy']:.1%}", flush=True,
            )
        else:
            print(event + ": " + json.dumps(details), flush=True)

    emit("run_started", {"run_id": run_id})
    write_json(directory / "config.json", c)
    snapshots = directory / "source"
    hashes = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        destination = snapshots / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = digest(destination)
    environment = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "cpu_threads": c["cpu_threads"],
        "logical_cpus": os.cpu_count(), "device": "cpu",
    }
    write_json(directory / "environment.json", environment)
    manifest = {
        "run_id": run_id, "source_sha256": hashes,
        "config_sha256": digest(directory / "config.json"),
        "environment_sha256": digest(directory / "environment.json"),
        "datasets_sha256": {},
    }
    results = []
    references = []
    try:
        # Freeze every dataset before any model sees training or evaluation data.
        worlds = {}
        data_dir = directory / "data"
        data_dir.mkdir()
        for seed in c["seeds"]:
            world = make_world(c, seed)
            data_path = data_dir / f"seed_{seed}.npz"
            save_world(data_path, world)
            worlds[seed] = world
            manifest["datasets_sha256"][str(data_path.relative_to(directory))] = digest(data_path)
        write_json(directory / "manifest.json", manifest)
        emit("inputs_frozen", {"manifest_sha256": digest(directory / "manifest.json")})
        for index, seed in enumerate(c["seeds"]):
            world = worlds[seed]
            seed_dir = directory / f"seed_{seed}"
            seed_dir.mkdir()
            # Rotate execution order to reduce systematic timing bias.
            methods = c["methods"][index % len(c["methods"]):] + c["methods"][:index % len(c["methods"])]
            for method in methods:
                emit("method_started", {"seed": seed, "method": method})
                result = fit_method(c, seed, method, world, seed_dir / method, emit, deadline)
                results.append(result)
                emit("method_completed", {
                    "seed": seed, "method": method,
                    "final_mean_accuracy": result["final_mean_accuracy"],
                    "training_seconds": result["training_seconds"],
                })
            reference, weights = fit_linear_reference(c, world)
            reference["seed"] = seed
            references.append(reference)
            write_json(seed_dir / "linear_reference.json", reference)
            with (seed_dir / "linear_weights.npy").open("xb") as f:
                np.save(f, weights, allow_pickle=False)
        expected_updates = c["tasks"] * c["steps_per_task"]
        for r in results:
            if r["updates"] != expected_updates or r["update_examples"] != expected_updates * c["batch_size"]:
                raise AssertionError("Training budgets differ")
        for seed in c["seeds"]:
            if len({r["initial_model_sha256"] for r in results if r["seed"] == seed}) != 1:
                raise AssertionError("Initial neural model weights differ")
        summary = {
            "run_id": run_id, "status": "completed",
            "wall_seconds": time.perf_counter() - started,
            "peak_process_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "decision": decision(results, c),
            "results": results, "linear_references": references,
        }
        write_json(directory / "summary.json", summary)
        artifact_hashes = {
            str(p.relative_to(directory)): digest(p)
            for p in directory.rglob("*")
            if p.is_file() and p.name != "events.jsonl"
        }
        write_json(directory / "artifacts.json", artifact_hashes)
        emit("run_completed", {
            "wall_seconds": summary["wall_seconds"],
            "verdict": summary["decision"]["verdict"],
            "artifacts_sha256": digest(directory / "artifacts.json"),
        })
        print("Saved run: " + str(directory), flush=True)
        return directory
    except BaseException as exc:
        failure = {
            "status": "failed" if not isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "incomplete",
            "error_type": type(exc).__name__, "error": str(exc),
            "completed_methods": len(results), "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(directory / "failure.json", failure)
        emit("run_stopped", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
