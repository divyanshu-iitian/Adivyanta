"""Export a compact trial checkpoint even if it was not validation-selected."""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=Path, required=True, help="Trainer's optimizer checkpoint")
    parser.add_argument("--template", type=Path, required=True, help="Compatible compact checkpoint with config")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    last = torch.load(args.last, map_location="cpu", weights_only=False)
    template = torch.load(args.template, map_location="cpu", weights_only=False)
    if set(last["model"]) != set(template["model"]) | {"head.weight"}:
        raise ValueError("Model state differs from template architecture")
    weights = {name: tensor.detach().cpu().half().contiguous()
               for name, tensor in last["model"].items() if name != "head.weight"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": weights, "config": template["config"], "step": last["step"],
                "val_loss": None, "parameters": template["parameters"],
                "selection": "unselected_trial_step"}, args.out)
    print(f"Saved unselected step {last['step']} to {args.out}")


if __name__ == "__main__":
    main()
