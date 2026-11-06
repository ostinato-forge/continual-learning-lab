"""Verify task partitions, stage predictions, and protected parameters."""

import json

import numpy as np
import torch

from continual_lab import Examples, ReplayMemory, accuracy, digest, model_digest, validate_config
from partition_models import architecture_stats, make_partition_model


def audit_stage_states(directory, c, architecture):
    stage_count = protection_checks = invariant_prediction_checks = 0
    stats = architecture_stats(c, architecture)
    for seed in c["seeds"]:
        folder = directory / f"seed_{seed}" / "online"
        result = json.loads((folder / "metrics.json").read_text())
        trace = json.loads((folder / "component_trace.json").read_text())
        states = torch.load(folder / "stage_states.pt", map_location="cpu", weights_only=True)
        if len(states) != c["tasks"] or len(trace["stages"]) != c["tasks"] or len(result["stages"]) != c["tasks"]:
            raise AssertionError("Task-end evidence is missing")
        if result["architecture"] != architecture or result["architecture_stats"] != stats:
            raise AssertionError("Architecture accounting differs")
        if result["parameter_count"] != stats["total_parameters"] or result["retained_examples"] != 0:
            raise AssertionError("Parameter or example-memory budget differs")
        if result["training_dense_forward_macs"] != result["update_examples"] * stats["dense_forward_macs_per_example"]:
            raise AssertionError("Forward operation accounting differs")
        model = make_partition_model(c, seed, architecture)
        previous_hashes = model.component_hashes()
        if trace["initial"] != previous_hashes or result["initial_model_sha256"] != model_digest(model):
            raise AssertionError("Initial components differ from the seeded model")
        with np.load(directory / "data" / f"seed_{seed}.npz", allow_pickle=False) as data:
            if result["order"] != data["order"].tolist():
                raise AssertionError("Recorded task order differs")

            def evaluate():
                model.eval()
                with torch.no_grad():
                    logits = [model(torch.from_numpy(data[f"test_{task}_x"])) for task in range(c["tasks"])]
                    scores = [float((value.argmax(dim=1) == torch.from_numpy(data[f"test_{task}_y"])).float().mean().item()) for task, value in enumerate(logits)]
                return logits, scores

            previous_logits, _ = evaluate()
            scores_by_stage = []
            for stage, task_value in enumerate(data["order"]):
                task = int(task_value)
                saved = states[stage]
                recorded = trace["stages"][stage]
                if (saved["stage"], saved["task"]) != (stage, task) or (recorded["stage"], recorded["task"]) != (stage, task):
                    raise AssertionError("A model state belongs to the wrong stage")
                model.load_state_dict(saved["model"])
                hashes = model.component_hashes()
                if hashes != recorded["hashes"] or hashes != result["stages"][stage]["component_hashes"]:
                    raise AssertionError("Component hashes do not match the saved model")
                logits, scores = evaluate()
                if not np.allclose(scores, result["stages"][stage]["test_accuracy_by_task"], rtol=0, atol=1e-7):
                    raise AssertionError("A task-end checkpoint does not reproduce reported scores")
                for other in range(c["tasks"]):
                    if other == task:
                        continue
                    if hashes[f"task_{other}"] != previous_hashes[f"task_{other}"]:
                        raise AssertionError("An inactive task component changed")
                    protection_checks += 1
                    if architecture["kind"] == "experts":
                        if not torch.equal(logits[other], previous_logits[other]):
                            raise AssertionError("Learning changed an independent task's predictions")
                        invariant_prediction_checks += 1
                previous_hashes, previous_logits = hashes, logits
                scores_by_stage.append(scores)
                stage_count += 1
            final = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
            if any(not torch.equal(value, final["model"][name]) for name, value in model.state_dict().items()):
                raise AssertionError("The last stage state differs from the final checkpoint")
            matrix = np.asarray(scores_by_stage)
            order = data["order"]
            acquired = np.array([matrix[i, task] for i, task in enumerate(order)])
            expected = {
                "final_mean_accuracy": float(matrix[-1].mean()),
                "mean_acquisition_accuracy": float(acquired.mean()),
                "old_task_retention_drop": float(np.mean(acquired[:-1] - matrix[-1, order[:-1]])),
            }
            if any(abs(result[key] - value) > 1e-7 for key, value in expected.items()):
                raise AssertionError("Derived metrics differ from task-end model predictions")
    return {
        "task_end_states_reproduced": stage_count,
        "inactive_component_checks": protection_checks,
        "independent_prediction_checks": invariant_prediction_checks,
    }


def verify_partition(directory):
    c = json.loads((directory / "config.json").read_text())
    validate_config(c)
    architecture = json.loads((directory / "architecture.json").read_text())
    if c["methods"] != ["online"] or architecture["kind"] not in ("heads", "experts"):
        raise ValueError("This audit expects a partitioned current-only condition")
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
                model = make_partition_model(c, seed, architecture)
                if model_digest(model) != metrics["initial_model_sha256"]:
                    raise AssertionError("Recorded initialization differs from the seeded architecture")
                model.load_state_dict(checkpoint["model"])
                actual = [accuracy(model, examples) for examples in testing]
                recorded = metrics["stages"][-1]["test_accuracy_by_task"]
                if not np.allclose(actual, recorded, rtol=0, atol=1e-7):
                    raise AssertionError("Checkpoint predictions differ from reported results")
                checked_models += 1
                checked_draws += draws.size
        if len(initial_hashes) != 1:
            raise AssertionError("Initial model weights were not paired")
    checked = {
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

    checked.update(audit_stage_states(directory, c, architecture))
    return checked
