"""Run one fixed adaptive replay policy and three paired reference methods."""

import argparse
import copy
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

from adaptive_controller import validate_policy
from adaptive_training import fit_with_policy
from continual_lab import digest, fit_linear_reference, fit_method, make_world, save_world, validate_config, write_json
from run_experiment import source_paths


ROOT = Path(__file__).resolve().parent


def validate_plan(plan, base):
    keys = {
        "experiment", "base_config", "seeds", "excluded_seeds", "memory_capacity",
        "fixed_replay_fraction", "controller", "final_accuracy_target", "retention_margin",
        "acquisition_tolerance", "required_successes", "wall_seconds_limit",
    }
    if set(plan) != keys:
        raise ValueError("Unexpected study settings")
    seeds = plan["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Use unique nonnegative seeds")
    if set(seeds) & (set(base["seeds"]) | set(plan["excluded_seeds"])):
        raise ValueError("The study must use fresh seeds")
    if type(plan["required_successes"]) is not int or not 1 <= plan["required_successes"] <= len(seeds):
        raise ValueError("Invalid success count")
    if not 0 < plan["final_accuracy_target"] <= 1:
        raise ValueError("Invalid final accuracy target")
    if not 0 <= plan["retention_margin"] <= 1 or not 0 <= plan["acquisition_tolerance"] <= 1:
        raise ValueError("Invalid comparison margins")
    if type(plan["wall_seconds_limit"]) is not int or plan["wall_seconds_limit"] < 1:
        raise ValueError("Invalid time limit")
    if not float(plan["fixed_replay_fraction"] * base["batch_size"]).is_integer():
        raise ValueError("Fixed replay must use whole example counts")
    validate_config(base)
    for part in ("baselines", "adaptive"):
        c = condition_config(plan, base, part)
        validate_config(c)
        validate_policy(plan["controller"], c)


def condition_config(plan, base, part):
    if part not in ("baselines", "adaptive"):
        raise ValueError("Unknown condition")
    c = copy.deepcopy(base)
    c.update(
        experiment=plan["experiment"] + "_" + part, seeds=list(plan["seeds"]),
        methods=["full_history", "online", "replay"] if part == "baselines" else ["replay"],
        memory_capacity=plan["memory_capacity"], wall_seconds_limit=plan["wall_seconds_limit"],
        replay_fraction=plan["fixed_replay_fraction"] if part == "baselines" else plan["controller"]["initial_fraction"],
        decision={"retention_margin": plan["retention_margin"], "acquisition_tolerance": plan["acquisition_tolerance"], "required_seed_wins": plan["required_successes"]},
    )
    return c


def observed_seeds():
    observed = set()
    for name in ("runs", "studies"):
        for path in (ROOT / name).rglob("config.json"):
            observed.update(json.loads(path.read_text())["seeds"])
    return observed


def run_study(plan_path, output_root):
    plan = json.loads(plan_path.read_text())
    base = json.loads((ROOT / plan["base_config"]).read_text())
    validate_plan(plan, base)
    seen = observed_seeds()
    if seen & set(plan["seeds"]):
        raise ValueError("At least one requested seed already has scored run evidence")
    torch.set_num_threads(base["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = output_root / (stamp + "-adaptive-" + uuid.uuid4().hex[:8])
    study.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + plan["wall_seconds_limit"]

    def emit(event, details):
        with (study / "events.jsonl").open("a") as f:
            f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}, allow_nan=False) + "\n")
        print(event + ": " + json.dumps(details), flush=True)

    emit("study_started", {"path": str(study)})
    write_json(study / "plan.json", plan)
    write_json(study / "base_config.json", base)
    write_json(study / "seed_audit.json", {"previously_observed": sorted(seen), "study_seeds_were_unobserved": True})
    environment = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "cpu_threads": base["cpu_threads"],
        "logical_cpus": os.cpu_count(), "device": "cpu",
    }
    write_json(study / "environment.json", environment)
    source_hashes = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        destination = study / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[str(relative)] = digest(destination)
    configs = {part: condition_config(plan, base, part) for part in ("baselines", "adaptive")}
    results = {part: [] for part in configs}
    references = []
    completed = 0
    try:
        worlds = {seed: make_world(configs["baselines"], seed) for seed in plan["seeds"]}
        child_hashes = {}
        for part, c in configs.items():
            child = study / part
            (child / "data").mkdir(parents=True)
            shutil.copytree(study / "source", child / "source")
            write_json(child / "config.json", c)
            write_json(child / "environment.json", environment)
            if part == "adaptive":
                write_json(child / "policy.json", plan["controller"])
            data_hashes = {}
            for seed, world in worlds.items():
                path = child / "data" / f"seed_{seed}.npz"
                save_world(path, world)
                data_hashes[str(path.relative_to(child))] = digest(path)
                (child / f"seed_{seed}").mkdir()
            write_json(child / "manifest.json", {
                "source_sha256": source_hashes, "config_sha256": digest(child / "config.json"),
                "environment_sha256": digest(child / "environment.json"), "datasets_sha256": data_hashes,
                "study_plan_sha256": digest(study / "plan.json"),
                "policy_sha256": digest(child / "policy.json") if part == "adaptive" else None,
            })
            child_hashes[part] = digest(child / "manifest.json")
        write_json(study / "manifest.json", {
            "plan_sha256": digest(study / "plan.json"), "base_sha256": digest(study / "base_config.json"),
            "source_sha256": source_hashes, "condition_manifest_sha256": child_hashes,
            "seed_audit_sha256": digest(study / "seed_audit.json"),
        })
        emit("inputs_frozen", {"manifest_sha256": digest(study / "manifest.json")})
        arms = [("baselines", "online"), ("baselines", "replay"), ("adaptive", "replay"), ("baselines", "full_history")]
        for index, seed in enumerate(plan["seeds"]):
            order = arms[index % len(arms):] + arms[:index % len(arms)]
            for part, method in order:
                child = study / part
                emit("method_started", {"seed": seed, "part": part, "method": method})

                def child_emit(event, details):
                    with (child / "events.jsonl").open("a") as f:
                        f.write(json.dumps({"event": event, **details}, allow_nan=False) + "\n")

                args = (configs[part], seed, method, worlds[seed], child / f"seed_{seed}" / method, child_emit, deadline)
                result = fit_with_policy(*args, policy=plan["controller"]) if part == "adaptive" else fit_method(*args)
                results[part].append(result)
                completed += 1
                emit("method_completed", {"seed": seed, "part": part, "method": method, "completed_models": completed})
            reference, weights = fit_linear_reference(configs["baselines"], worlds[seed])
            reference["seed"] = seed
            references.append(reference)
            for part in configs:
                folder = study / part / f"seed_{seed}"
                write_json(folder / "linear_reference.json", reference)
                with (folder / "linear_weights.npy").open("xb") as f:
                    np.save(f, weights, allow_pickle=False)
        receipts = {}
        for part in configs:
            child = study / part
            write_json(child / "summary.json", {
                "status": "completed", "results": results[part], "linear_references": references,
                "summed_method_wall_seconds": sum(r["method_wall_seconds"] for r in results[part]),
            })
            artifacts = {
                str(path.relative_to(child)): digest(path) for path in child.rglob("*")
                if path.is_file() and path.name != "events.jsonl"
            }
            write_json(child / "artifacts.json", artifacts)
            receipts[part] = digest(child / "artifacts.json")
        summary = {
            "status": "completed", "neural_models": completed,
            "wall_seconds": time.perf_counter() - started,
            "peak_process_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "manifest_sha256": digest(study / "manifest.json"), "condition_artifacts_sha256": receipts,
        }
        write_json(study / "study_summary.json", summary)
        emit("study_completed", summary)
        return study
    except BaseException as exc:
        failure = {
            "status": "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed",
            "error_type": type(exc).__name__, "error": str(exc), "completed_models": completed,
            "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter() - started,
        }
        write_json(study / "failure.json", failure)
        emit("study_stopped", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/adaptive_replay.json")
    parser.add_argument("--output", type=Path, default=ROOT / "studies")
    args = parser.parse_args()
    run_study(args.plan.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
