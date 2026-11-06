"""Train the fixed partitioned architectures using the original learning loop.

The shared architecture path is checked against the unchanged baseline code.
Inactive-gradient checks and component snapshots support evidence verification.
"""

import time

import numpy as np
import torch
from torch import nn

from continual_lab import ReplayMemory, accuracy, model_digest, pack, training_batch, write_json
from partition_models import architecture_stats, make_partition_model


def fit_partition(c, seed, method, world, output, emit, deadline, architecture):
    if architecture["kind"] != "shared" and method != "online":
        raise ValueError("Partitioned conditions use current examples only")
    output.mkdir()
    model = make_partition_model(c, seed, architecture)
    stats = architecture_stats(c, architecture)
    partitioned = architecture["kind"] != "shared"
    initial_components = model.component_hashes() if partitioned else {}
    component_records = []
    stage_states = []
    initial_hash = model_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"]
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 300]))
    memory = ReplayMemory(c["memory_capacity"], c["features"] + c["tasks"], seed + 400)
    matrix = []
    curves = []
    batches = []
    stage_records = []
    archive = []
    seen_tasks = set()
    updates = 0
    exposures = 0
    training_seconds = 0.0
    evaluation_seconds = 0.0
    memory_seconds = 0.0
    evidence_seconds = 0.0
    started = time.perf_counter()
    for stage, task_value in enumerate(world.order):
        task = int(task_value)
        current = world.train[task]
        seen_tasks.add(task)
        # The full archive belongs only to the full-history method.
        if method == "full_history":
            archive.append(current)
            history = pack(archive)
        else:
            history = current
        available_ids = set(current.ids.tolist())
        if method == "full_history":
            available_ids.update(history.ids.tolist())
        elif method == "replay":
            available_ids.update(memory.examples().ids.tolist())
        unique_current = set()
        stage_training_seconds = 0.0
        sampled_tasks = {str(t): 0 for t in sorted(seen_tasks)}
        for step in range(c["steps_per_task"] + 1):
            if time.perf_counter() > deadline:
                raise TimeoutError("The frozen wall-clock limit was reached")
            if step in c["curve_steps"]:
                t0 = time.perf_counter()
                val_accuracy = accuracy(model, world.validation[task])
                evaluation_seconds += time.perf_counter() - t0
                curves.append({
                    "seed": seed, "method": method, "stage": stage, "task": task,
                    "step": step, "validation_accuracy": val_accuracy,
                    "unique_current_examples_used": len(unique_current),
                    "current_examples_available": len(current.ids),
                    "total_update_examples": updates * c["batch_size"],
                    "stage_training_seconds": stage_training_seconds,
                })
            if step == c["steps_per_task"]:
                break
            t0 = time.perf_counter()
            x, y, ids, task_ids = training_batch(method, current, history, memory, c, rng)
            if not set(ids.tolist()) <= available_ids or not set(task_ids.tolist()) <= seen_tasks:
                raise AssertionError("Training accessed unavailable data")
            batches.append(ids)
            unique_current.update(ids[task_ids == task].tolist())
            for seen in sampled_tasks:
                sampled_tasks[seen] += int(np.count_nonzero(task_ids == int(seen)))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            loss.backward()
            if partitioned:
                audit_started = time.perf_counter()
                model.check_inactive_gradients(task)
                audit_seconds = time.perf_counter() - audit_started
                evidence_seconds += audit_seconds
                t0 += audit_seconds
            optimizer.step()
            elapsed = time.perf_counter() - t0
            training_seconds += elapsed
            stage_training_seconds += elapsed
            updates += 1
            exposures += len(ids)
        t0 = time.perf_counter()
        scores = [accuracy(model, world.test[t]) for t in range(c["tasks"])]
        evaluation_seconds += time.perf_counter() - t0
        matrix.append(scores)
        t0 = time.perf_counter()
        if method == "replay":
            memory.add(current)
        memory_seconds += time.perf_counter() - t0
        past_count = (
            sum(len(e.ids) for e in archive[:-1])
            if method == "full_history" else
            min(c["memory_capacity"], stage * c["train_examples"])
            if method == "replay" else 0
        )
        retained_count = (
            sum(len(e.ids) for e in archive)
            if method == "full_history" else memory.size if method == "replay" else 0
        )
        record = {
            "seed": seed, "method": method, "stage": stage, "task": task,
            "test_accuracy_by_task": scores,
            "current_test_accuracy": scores[task],
            "mean_seen_test_accuracy": float(np.mean([scores[t] for t in seen_tasks])),
            "unique_current_examples_used": len(unique_current),
            "sampled_examples_by_task": sampled_tasks,
            "past_memory_examples_during_stage": past_count,
            "retained_examples_after_stage": retained_count,
            "stage_training_seconds": stage_training_seconds,
        }
        if partitioned:
            capture_started = time.perf_counter()
            components = model.component_hashes()
            previous = component_records[-1]["hashes"] if component_records else initial_components
            for other in range(c["tasks"]):
                if other != task and components[f"task_{other}"] != previous[f"task_{other}"]:
                    raise AssertionError("Learning changed an inactive task component")
            component_records.append({"stage": stage, "task": task, "hashes": components})
            record["component_hashes"] = components
            stage_states.append({"stage": stage, "task": task, "model": {name: value.detach().clone() for name, value in model.state_dict().items()}})
            evidence_seconds += time.perf_counter() - capture_started
        stage_records.append(record)
        emit("stage_completed", record)
    matrix = np.asarray(matrix)
    acquired = np.array([matrix[i, int(t)] for i, t in enumerate(world.order)])
    final = matrix[-1]
    old_tasks = world.order[:-1]
    retention_drop = float(np.mean(acquired[:-1] - final[old_tasks]))
    target_hits = []
    for stage in range(c["tasks"]):
        hits = [row for row in curves if row["stage"] == stage and row["validation_accuracy"] >= c["target_accuracy"]]
        target_hits.append({
            "stage": stage,
            "first_recorded_step": hits[0]["step"] if hits else None,
            "unique_examples_at_first_hit": hits[0]["unique_current_examples_used"] if hits else None,
        })
    result = {
        "seed": seed, "method": method, "order": world.order.tolist(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_model_sha256": initial_hash,
        "final_mean_accuracy": float(final.mean()),
        "mean_acquisition_accuracy": float(acquired.mean()),
        "old_task_retention_drop": retention_drop,
        "updates": updates, "update_examples": exposures,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "memory_maintenance_seconds": memory_seconds,
        "method_wall_seconds": time.perf_counter() - started,
        "retained_examples": stage_records[-1]["retained_examples_after_stage"],
        "retained_example_payload_bytes": stage_records[-1]["retained_examples_after_stage"] * (
            (c["features"] + c["tasks"]) * 4 + 8 + 8 + 8
        ),
        "target_hits": target_hits,
        "stages": stage_records,
        "curves": curves,
    }
    capture_started = time.perf_counter()
    if partitioned:
        write_json(output / "component_trace.json", {"initial": initial_components, "stages": component_records})
        with (output / "stage_states.pt").open("xb") as f:
            torch.save(stage_states, f)
    evidence_seconds += time.perf_counter() - capture_started
    result["architecture"] = architecture
    result["architecture_stats"] = stats
    result["training_dense_forward_macs"] = exposures * stats["dense_forward_macs_per_example"]
    result["evidence_seconds"] = evidence_seconds
    result["method_wall_seconds"] = time.perf_counter() - started
    write_json(output / "metrics.json", result)
    with (output / "batch_ids.npy").open("xb") as f:
        np.save(f, np.stack(batches), allow_pickle=False)
    with (output / "checkpoint.pt").open("xb") as f:
        torch.save({
            "model": model.state_dict(), "seed": seed, "method": method,
            "config": c, "initial_model_sha256": initial_hash,
        }, f)
    return result

