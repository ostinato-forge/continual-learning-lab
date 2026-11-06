"""Change replay allocation using retained training examples only."""

import time

import numpy as np

from continual_lab import Examples, accuracy


def validate_policy(policy, c):
    keys = {"fractions", "initial_fraction", "probe_examples", "probe_interval", "increase_drop", "decrease_drop"}
    if set(policy) != keys:
        raise ValueError("Unexpected controller settings")
    fractions = policy["fractions"]
    if not fractions or any(type(v) not in (int, float) or not 0 < v < 1 for v in fractions):
        raise ValueError("Replay shares must lie between zero and one")
    if fractions != sorted(set(fractions)) or policy["initial_fraction"] not in fractions:
        raise ValueError("Replay shares must be ordered, unique, and include the initial share")
    if any(not float(v * c["batch_size"]).is_integer() for v in fractions):
        raise ValueError("Each share must give a whole number of batch examples")
    for key in ("probe_examples", "probe_interval"):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError("Probe size and interval must be positive integers")
    if not 0 <= policy["decrease_drop"] < policy["increase_drop"] <= 1:
        raise ValueError("The two switching thresholds must have a gap")
    if c["memory_capacity"] < 1:
        raise ValueError("Adaptive replay requires memory")


def next_fraction(fraction, reference, observed, policy):
    levels = policy["fractions"]
    index = levels.index(fraction)
    drop = reference - observed
    if drop >= policy["increase_drop"]:
        index = min(index + 1, len(levels) - 1)
    elif drop <= policy["decrease_drop"]:
        index = max(index - 1, 0)
    return levels[index]


class AdaptiveController:
    def __init__(self, policy, seed):
        self.policy = policy
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 500]))
        self.trace = []
        self.states = []
        self.seconds = 0.0
        self.evidence_seconds = 0.0
        self.forward_examples = 0

    def begin_stage(self, examples, stage):
        started = time.perf_counter()
        if examples.split != "train":
            raise ValueError("The controller can only probe retained training examples")
        self.stage = stage
        self.reference = None
        self.fraction = self.policy["initial_fraction"]
        self.probe = None
        if len(examples.ids):
            indexes = self.rng.choice(len(examples.ids), size=min(self.policy["probe_examples"], len(examples.ids)), replace=False)
            self.probe = Examples(
                examples.x[indexes], examples.y[indexes], examples.ids[indexes].copy(),
                examples.task_ids[indexes].copy(), "train",
            )
        self.seconds += time.perf_counter() - started

    def before_update(self, model, step):
        started = time.perf_counter()
        if self.probe is None:
            self.seconds += time.perf_counter() - started
            return 0.0
        due = step % self.policy["probe_interval"] == 0
        if due:
            observed = accuracy(model, self.probe)
            previous = self.fraction
            if self.reference is None:
                self.reference = observed
            else:
                self.fraction = next_fraction(previous, self.reference, observed, self.policy)
            self.forward_examples += len(self.probe.ids)
        self.seconds += time.perf_counter() - started
        if due:
            capture_started = time.perf_counter()
            self.trace.append({
                "stage": self.stage, "step": step, "ids": self.probe.ids.tolist(),
                "accuracy": observed, "reference_accuracy": self.reference,
                "fraction_before": previous, "fraction_after": self.fraction,
            })
            self.states.append({
                "stage": self.stage, "step": step,
                "model": {name: value.detach().clone() for name, value in model.state_dict().items()},
            })
            self.evidence_seconds += time.perf_counter() - capture_started
        return self.fraction
