"""Networks with supplied task routing and protected task-specific parameters."""

import torch
from torch import nn

from continual_lab import make_model, model_digest


def architecture_stats(c, architecture):
    kind = architecture["kind"]
    if kind == "shared":
        if set(architecture) != {"kind"}:
            raise ValueError("Unexpected shared architecture settings")
        a = b = c["hidden_width"]
        inputs = c["features"] + c["tasks"]
        total = (inputs + 1) * a + (a + 1) * b + (b + 1) * c["classes"]
        active = total
    else:
        if kind not in ("heads", "experts") or set(architecture) != {"kind", "widths"}:
            raise ValueError("Unknown architecture settings")
        widths = architecture["widths"]
        if len(widths) != 2 or any(type(v) is not int or v < 1 for v in widths):
            raise ValueError("Use two positive integer hidden widths")
        a, b = widths
        inputs = c["features"] + c["tasks"] if kind == "heads" else c["features"]
        trunk = (inputs + 1) * a + (a + 1) * b
        head = (b + 1) * c["classes"]
        active = trunk + head
        total = trunk + c["tasks"] * head if kind == "heads" else c["tasks"] * active
    return {
        "total_parameters": total,
        "active_parameters_per_example": active,
        "dense_forward_macs_per_example": inputs * a + a * b + b * c["classes"],
    }


class RoutedNetwork(nn.Module):
    def __init__(self, c, architecture):
        super().__init__()
        self.features, self.tasks, self.classes = c["features"], c["tasks"], c["classes"]
        self.kind = architecture["kind"]
        a, b = architecture["widths"]
        if self.kind == "heads":
            self.trunk = nn.Sequential(nn.Linear(self.features + self.tasks, a), nn.ReLU(), nn.Linear(a, b), nn.ReLU())
            self.parts = nn.ModuleList([nn.Linear(b, self.classes) for _ in range(self.tasks)])
        else:
            self.parts = nn.ModuleList([
                nn.Sequential(nn.Linear(self.features, a), nn.ReLU(), nn.Linear(a, b), nn.ReLU(), nn.Linear(b, self.classes))
                for _ in range(self.tasks)
            ])

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.features + self.tasks:
            raise ValueError("Inputs must include the supplied task identity")
        identity = x[:, self.features:]
        if not bool(torch.all((identity == 0) | (identity == 1))) or not bool(torch.all(identity.sum(dim=1) == 1)):
            raise ValueError("Each input must identify exactly one known task")
        tasks = identity.argmax(dim=1)
        hidden = self.trunk(x) if self.kind == "heads" else x[:, :self.features]
        output = x.new_zeros((len(x), self.classes))
        for task, part in enumerate(self.parts):
            indexes = torch.nonzero(tasks == task, as_tuple=False).flatten()
            if indexes.numel():
                output = output.index_copy(0, indexes, part(hidden.index_select(0, indexes)))
        return output

    def component_hashes(self):
        result = {f"task_{i}": model_digest(part) for i, part in enumerate(self.parts)}
        if self.kind == "heads":
            result["shared"] = model_digest(self.trunk)
        return result

    def check_inactive_gradients(self, current_task):
        for task, part in enumerate(self.parts):
            if task != current_task and any(p.grad is not None for p in part.parameters()):
                raise AssertionError("An inactive task component received a gradient")


def make_partition_model(c, seed, architecture):
    stats = architecture_stats(c, architecture)
    if architecture["kind"] == "shared":
        model = make_model(c, seed)
    else:
        torch.manual_seed(seed)
        model = RoutedNetwork(c, architecture)
    if sum(p.numel() for p in model.parameters()) != stats["total_parameters"]:
        raise AssertionError("Actual parameter count differs from the architecture formula")
    return model
