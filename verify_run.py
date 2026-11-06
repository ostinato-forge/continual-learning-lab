"""Verify saved evidence, training access, budgets, and checkpoint predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from continual_lab import (
    Examples, ReplayMemory, accuracy, digest, make_model, validate_config, write_json,
)


def verify(directory):
    c = json.loads((directory / "config.json").read_text())
    validate_config(c)
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "completed":
        raise AssertionError("The run did not complete")
    artifacts = json.loads((directory / "artifacts.json").read_text())
    for relative, expected in artifacts.items():
        if digest(directory / relative) != expected:
            raise AssertionError(f"Artifact hash mismatch: {relative}")
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if digest(directory / "source" / relative) != expected:
            raise AssertionError(f"Source hash mismatch: {relative}")
    if digest(directory / "config.json") != manifest["config_sha256"]:
        raise AssertionError("Config differs from frozen manifest")
    torch.set_num_threads(c["cpu_threads"])
    expected_rows = c["tasks"] * c["steps_per_task"]
    if len(summary["results"]) != len(c["seeds"]) * len(c["methods"]):
        raise AssertionError("Missing method results")
    checked_models = 0
    checked_draws = 0
    for seed in c["seeds"]:
        initial_hashes = set()
        with np.load(directory / "data" / f"seed_{seed}.npz", allow_pickle=False) as data:
            training = []
            testing = []
            for task in range(c["tasks"]):
                training.append(Examples(
                    torch.from_numpy(data[f"train_{task}_x"]),
                    torch.from_numpy(data[f"train_{task}_y"]),
                    data[f"train_{task}_ids"],
                    np.full(c["train_examples"], task, dtype=np.int64), "train",
                ))
                testing.append(Examples(
                    torch.from_numpy(data[f"test_{task}_x"]),
                    torch.from_numpy(data[f"test_{task}_y"]),
                    data[f"test_{task}_ids"],
                    np.full(c["test_examples"], task, dtype=np.int64), "test",
                ))
            for method in c["methods"]:
                folder = directory / f"seed_{seed}" / method
                metrics = json.loads((folder / "metrics.json").read_text())
                result = next(r for r in summary["results"] if r["seed"] == seed and r["method"] == method)
                if metrics != result:
                    raise AssertionError("Summary differs from per-method metrics")
                initial_hashes.add(metrics["initial_model_sha256"])
                draws = np.load(folder / "batch_ids.npy", allow_pickle=False)
                if draws.shape != (expected_rows, c["batch_size"]):
                    raise AssertionError("Training draw count differs from budget")
                if metrics["updates"] != expected_rows or metrics["update_examples"] != draws.size:
                    raise AssertionError("Recorded training cost differs from draw trace")
                reservoir = ReplayMemory(
                    c["memory_capacity"], c["features"] + c["tasks"], seed + 400
                )
                history = set()
                for stage, task in enumerate(data["order"]):
                    current = training[int(task)]
                    history.update(current.ids.tolist())
                    allowed = (
                        history if method == "full_history" else
                        set(current.ids.tolist()) | set(reservoir.examples().ids.tolist())
                        if method == "replay" else set(current.ids.tolist())
                    )
                    block = draws[
                        stage * c["steps_per_task"]:(stage + 1) * c["steps_per_task"]
                    ]
                    if not set(block.ravel().tolist()) <= allowed:
                        raise AssertionError("Training used future, evaluation, or unretained examples")
                    if method == "replay":
                        reservoir.add(current)
                checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                model = make_model(c, seed)
                model.load_state_dict(checkpoint["model"])
                actual = [accuracy(model, examples) for examples in testing]
                recorded = metrics["stages"][-1]["test_accuracy_by_task"]
                if not np.allclose(actual, recorded, rtol=0, atol=1e-7):
                    raise AssertionError("Checkpoint predictions differ from reported results")
                checked_models += 1
                checked_draws += draws.size
        if len(initial_hashes) != 1:
            raise AssertionError("Initial model weights were not paired")
    return {
        "status": "passed",
        "artifact_files_checked": len(artifacts),
        "checkpoints_evaluated": checked_models,
        "training_example_draws_checked": checked_draws,
        "checks": [
            "Saved artifact and source hashes match",
            "Neural training budgets match the frozen configuration",
            "Neural initial weights are identical within each seed",
            "Every sampled example was available under its method",
            "No evaluation or future-task examples entered training",
            "Final checkpoint predictions reproduce the recorded test scores",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run.resolve())
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
