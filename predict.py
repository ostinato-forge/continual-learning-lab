"""Inspect a trained machine prediction on an input you choose."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from continual_lab import make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--method", choices=["full_history", "online", "replay"], default="replay")
    parser.add_argument("--machine", type=int, default=0)
    parser.add_argument("--values", type=float, nargs="+", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.run / f"seed_{args.seed}" / args.method / "checkpoint.pt",
        weights_only=True, map_location="cpu",
    )
    c = checkpoint["config"]
    if len(args.values) != c["features"] or not 0 <= args.machine < c["tasks"]:
        parser.error(f"Supply {c['features']} values and a machine from 0 to {c['tasks'] - 1}")
    if any(not np.isfinite(v) or abs(v) > 1 for v in args.values):
        parser.error("Values must be finite and between -1 and 1")
    torch.set_num_threads(c["cpu_threads"])
    model = make_model(c, args.seed)
    model.load_state_dict(checkpoint["model"])
    identity = [float(i == args.machine) for i in range(c["tasks"])]
    x = torch.tensor([args.values + identity], dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0].tolist()
    with np.load(args.run / "data" / f"seed_{args.seed}.npz", allow_pickle=False) as data:
        scores = np.asarray(args.values, dtype=np.float32) @ data["weights"][args.machine].T
        raw_bin = int(scores[0] > 0) + 2 * int(scores[1] > 0)
        target = int(data["label_maps"][args.machine][raw_bin])
    prediction = int(np.argmax(probabilities))
    print(json.dumps({
        "machine": args.machine, "method": args.method,
        "predicted_bin": prediction, "correct_bin": target,
        "correct": prediction == target, "probabilities": probabilities,
        "updates_performed": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
