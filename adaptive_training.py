"""Extend the original training loop with an optional replay controller.

The fixed-policy path is checked against the original implementation. The
original learning module remains unchanged for earlier study readback.
"""

import time

import numpy as np
import torch
from torch import nn

from adaptive_controller import AdaptiveController, validate_policy
from continual_lab import (
    ReplayMemory, accuracy, make_model, model_digest, pack, training_batch, write_json,
)


def fit_with_policy(c, seed, method, world, output, emit, deadline, policy=None):
    if policy is not None:
        validate_policy(policy, c)
        if method != "replay":
            raise ValueError("Only replay can use a controller")
    output.mkdir()
    model = make_model(c, seed)
    initial_hash = model_digest(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"]
    )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 300]))
    memory = ReplayMemory(c["memory_capacity"], c["features"] + c["tasks"], seed + 400)
    controller = AdaptiveController(policy, seed) if policy is not None else None
    replay_counts = []
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
    started = time.perf_counter()
    for stage, task_value in enumerate(world.order):
        task = int(task_value)
        current = world.train[task]
        if controller is not None:
            controller.begin_stage(memory.examples(), stage)
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
            fraction = controller.before_update(model, step) if controller is not None else c["replay_fraction"]
            t0 = time.perf_counter()
            batch_config = {**c, "replay_fraction": fraction}
            x, y, ids, task_ids = training_batch(method, current, history, memory, batch_config, rng)
            replay_counts.append(int(c["batch_size"] * fraction) if method == "replay" and memory.size else 0)
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
    if controller is not None:
        capture_started = time.perf_counter()
        write_json(output / "controller_trace.json", controller.trace)
        with (output / "replay_counts.npy").open("xb") as f:
            np.save(f, np.asarray(replay_counts, dtype=np.int64), allow_pickle=False)
        with (output / "probe_states.pt").open("xb") as f:
            torch.save(controller.states, f)
        controller.evidence_seconds += time.perf_counter() - capture_started
        result["controller"] = {
            "policy": policy, "probe_calls": len(controller.trace),
            "forward_examples": controller.forward_examples,
            "controller_seconds": controller.seconds,
            "evidence_seconds": controller.evidence_seconds,
            "replayed_examples": sum(replay_counts),
            "later_batch_mean_fraction": float(np.mean(replay_counts[c["steps_per_task"]:]) / c["batch_size"]),
        }
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

