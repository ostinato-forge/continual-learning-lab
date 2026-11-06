"""Small, supervised experiments in learning and retention."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


METHODS = ("full_history", "online", "replay")


def validate_config(c):
    required = {
        "experiment", "seeds", "methods", "tasks", "features", "classes",
        "train_examples", "validation_examples", "test_examples",
        "hidden_width", "steps_per_task", "batch_size", "learning_rate",
        "weight_decay", "memory_capacity", "replay_fraction", "curve_steps",
        "target_accuracy", "ridge_penalty", "cpu_threads",
        "wall_seconds_limit", "decision",
    }
    if set(c) != required:
        raise ValueError(f"Config keys differ: {set(c) ^ required}")
    for key in (
        "tasks", "features", "train_examples", "validation_examples",
        "test_examples", "hidden_width", "steps_per_task", "batch_size",
        "cpu_threads", "wall_seconds_limit",
    ):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["tasks"] < 2 or c["features"] < 2 or c["classes"] != 4:
        raise ValueError("Use at least two tasks and features, and four classes")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if any(type(s) is not int or s < 0 for s in c["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if not c["methods"] or len(set(c["methods"])) != len(c["methods"]):
        raise ValueError("Methods must be nonempty and unique")
    if not set(c["methods"]) <= set(METHODS):
        raise ValueError("Unknown learning method")
    if type(c["memory_capacity"]) is not int or c["memory_capacity"] < 0:
        raise ValueError("Memory capacity must be a nonnegative integer")
    if not 0 <= c["replay_fraction"] < 1:
        raise ValueError("Replay fraction must be in [0, 1)")
    if not 0 < c["target_accuracy"] <= 1:
        raise ValueError("Target accuracy must be in (0, 1]")
    if c["learning_rate"] <= 0 or c["weight_decay"] < 0 or c["ridge_penalty"] <= 0:
        raise ValueError("Invalid optimizer or ridge setting")
    if (
        sorted(set(c["curve_steps"])) != c["curve_steps"]
        or not c["curve_steps"]
        or c["curve_steps"][0] != 0
        or c["curve_steps"][-1] != c["steps_per_task"]
        or any(type(s) is not int or s < 0 for s in c["curve_steps"])
    ):
        raise ValueError("Curve steps must be unique, ordered, and span the training block")
    decision = c["decision"]
    if set(decision) != {"retention_margin", "acquisition_tolerance", "required_seed_wins"}:
        raise ValueError("Invalid decision settings")
    if not 0 <= decision["retention_margin"] <= 1 or not 0 <= decision["acquisition_tolerance"] <= 1:
        raise ValueError("Decision margins must be in [0, 1]")
    if not 1 <= decision["required_seed_wins"] <= len(c["seeds"]):
        raise ValueError("Required seed wins must fit the seed count")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


@dataclass
class Examples:
    x: torch.Tensor
    y: torch.Tensor
    ids: np.ndarray
    task_ids: np.ndarray
    split: str


@dataclass
class World:
    train: list
    validation: list
    test: list
    weights: np.ndarray
    label_maps: np.ndarray
    order: np.ndarray


def pack(examples):
    if not examples or any(e.split != "train" for e in examples):
        raise ValueError("Only training examples may enter a training pool")
    return Examples(
        torch.cat([e.x for e in examples]),
        torch.cat([e.y for e in examples]),
        np.concatenate([e.ids for e in examples]),
        np.concatenate([e.task_ids for e in examples]),
        "train",
    )


def make_world(c, seed):
    rules = np.random.default_rng(np.random.SeedSequence([seed, 100]))
    weights = []
    for _ in range(c["tasks"]):
        q, _ = np.linalg.qr(rules.normal(size=(c["features"], 2)))
        weights.append(q.T.astype(np.float32))
    weights = np.stack(weights)
    label_maps = np.stack([rules.permutation(4) for _ in range(c["tasks"])])
    order = np.random.default_rng(np.random.SeedSequence([seed, 101])).permutation(c["tasks"])
    splits = {"train": [], "validation": [], "test": []}
    offset = 0
    rows = set()
    for split_number, (name, groups) in enumerate(splits.items()):
        for task in range(c["tasks"]):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 200, split_number, task]))
            raw = rng.uniform(-1, 1, (c[name + "_examples"], c["features"])).astype(np.float32)
            for row in raw:
                key = row.tobytes()
                if key in rows:
                    raise ValueError("Duplicate raw input across data splits")
                rows.add(key)
            scores = raw @ weights[task].T
            bins = (scores[:, 0] > 0).astype(np.int64) + 2 * (scores[:, 1] > 0)
            labels = label_maps[task][bins]
            identity = np.zeros((len(raw), c["tasks"]), dtype=np.float32)
            identity[:, task] = 1
            groups.append(Examples(
                torch.from_numpy(np.concatenate([raw, identity], axis=1)),
                torch.from_numpy(labels.copy()),
                np.arange(offset, offset + len(raw), dtype=np.int64),
                np.full(len(raw), task, dtype=np.int64),
                name,
            ))
            offset += len(raw)
    return World(**splits, weights=weights, label_maps=label_maps, order=order)


def save_world(path, world):
    arrays = {"weights": world.weights, "label_maps": world.label_maps, "order": world.order}
    for name in ("train", "validation", "test"):
        for task, examples in enumerate(getattr(world, name)):
            arrays[f"{name}_{task}_x"] = examples.x.numpy()
            arrays[f"{name}_{task}_y"] = examples.y.numpy()
            arrays[f"{name}_{task}_ids"] = examples.ids
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def make_model(c, seed):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(c["features"] + c["tasks"], c["hidden_width"]),
        nn.ReLU(),
        nn.Linear(c["hidden_width"], c["hidden_width"]),
        nn.ReLU(),
        nn.Linear(c["hidden_width"], c["classes"]),
    )


@torch.no_grad()
def accuracy(model, examples):
    model.eval()
    prediction = model(examples.x).argmax(dim=1)
    return float((prediction == examples.y).float().mean().item())


class ReplayMemory:
    """Uniform reservoir sampling over previously supplied training examples."""

    def __init__(self, capacity, dimension, seed):
        self.capacity = capacity
        self.x = torch.empty((capacity, dimension), dtype=torch.float32)
        self.y = torch.empty(capacity, dtype=torch.int64)
        self.ids = np.empty(capacity, dtype=np.int64)
        self.task_ids = np.empty(capacity, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.seen = 0

    def add(self, examples):
        if examples.split != "train":
            raise ValueError("Evaluation examples cannot enter replay memory")
        for i in range(len(examples.ids)):
            self.seen += 1
            slot = self.seen - 1 if self.size < self.capacity else int(self.rng.integers(self.seen))
            if slot < self.capacity:
                self.x[slot].copy_(examples.x[i])
                self.y[slot] = examples.y[i]
                self.ids[slot] = examples.ids[i]
                self.task_ids[slot] = examples.task_ids[i]
                self.size = min(self.capacity, self.size + 1)

    def examples(self):
        return Examples(
            self.x[:self.size], self.y[:self.size], self.ids[:self.size],
            self.task_ids[:self.size], "train",
        )


def sample(examples, count, rng):
    if examples.split != "train":
        raise ValueError("Evaluation data cannot be sampled for training")
    indexes = rng.integers(len(examples.ids), size=count)
    return examples.x[indexes], examples.y[indexes], examples.ids[indexes], examples.task_ids[indexes]


def training_batch(method, current, history, memory, c, rng):
    if method == "full_history":
        return sample(history, c["batch_size"], rng)
    replay_count = (
        int(c["batch_size"] * c["replay_fraction"])
        if method == "replay" and memory.size else 0
    )
    fresh = sample(current, c["batch_size"] - replay_count, rng)
    if not replay_count:
        return fresh
    old = sample(memory.examples(), replay_count, rng)
    return (
        torch.cat([fresh[0], old[0]]), torch.cat([fresh[1], old[1]]),
        np.concatenate([fresh[2], old[2]]), np.concatenate([fresh[3], old[3]]),
    )


def fit_method(c, seed, method, world, output, emit, deadline):
    output.mkdir()
    model = make_model(c, seed)
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
    write_json(output / "metrics.json", result)
    with (output / "batch_ids.npy").open("xb") as f:
        np.save(f, np.stack(batches), allow_pickle=False)
    with (output / "checkpoint.pt").open("xb") as f:
        torch.save({
            "model": model.state_dict(), "seed": seed, "method": method,
            "config": c, "initial_model_sha256": initial_hash,
        }, f)
    return result


def fit_linear_reference(c, world):
    """Fit one separate linear classifier per known machine."""
    started = time.perf_counter()
    models = []
    scores = []
    for train, test in zip(world.train, world.test):
        x = np.column_stack([train.x.numpy()[:, :c["features"]], np.ones(len(train.ids))])
        y = np.eye(c["classes"])[train.y.numpy()]
        penalty = np.eye(x.shape[1]) * c["ridge_penalty"]
        penalty[-1, -1] = 0
        weight = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        tx = np.column_stack([test.x.numpy()[:, :c["features"]], np.ones(len(test.ids))])
        scores.append(float(np.mean((tx @ weight).argmax(axis=1) == test.y.numpy())))
        models.append(weight)
    return {
        "method": "separate_linear_reference",
        "final_mean_accuracy": float(np.mean(scores)),
        "test_accuracy_by_task": scores,
        "parameter_count": c["tasks"] * (c["features"] + 1) * c["classes"],
        "wall_seconds": time.perf_counter() - started,
        "training_examples": c["tasks"] * c["train_examples"],
        "budget_matched_to_neural_methods": False,
    }, np.stack(models)
