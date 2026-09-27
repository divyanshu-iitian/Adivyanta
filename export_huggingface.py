"""Build a standalone Hugging Face model repo from local scratch checkpoints."""
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "adivyanta_indic" / "expanded"
PACKAGE = ROOT / "data" / "huggingface_upload"


def convert(checkpoint, output):
    state = torch.load(SOURCE / checkpoint, map_location="cpu", weights_only=False)
    if state["parameters"] != 46_349_312:
        raise ValueError("Unexpected parameter count")
    tensors = {key: value.contiguous() for key, value in state["model"].items() if key != "head.weight"}
    save_file(tensors, PACKAGE / output, metadata={"format": "pt", "source_checkpoint": checkpoint})
    return state["config"]


def main():
    PACKAGE.mkdir(parents=True, exist_ok=True)
    base = convert("best.pt", "model.safetensors")
    chat = convert("chat_best.pt", "chat_model.safetensors")
    if base != chat:
        raise ValueError("Checkpoint configs differ")
    (PACKAGE / "config.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    for source, target in ((SOURCE / "tokenizer.json", "tokenizer.json"),
                           (ROOT / "model.py", "model.py"),
                           (ROOT / "huggingface" / "inference.py", "inference.py"),
                           (ROOT / "huggingface" / "README.md", "README.md"),
                           (ROOT / "huggingface" / "requirements.txt", "requirements.txt"),
                           (ROOT / "LICENSE-CODE", "LICENSE")):
        shutil.copy2(source, PACKAGE / target)
    for group, files in {
        "metrics": [SOURCE / "training_metrics.json", SOURCE / "chat_training_metrics.json",
                    SOURCE / "base_test_metrics.json", SOURCE / "test_metrics.json",
                    ROOT / "benchmarks" / "math_summary.json"],
        "benchmarks": [ROOT / "benchmarks" / "indic_expanded_base_outputs.jsonl",
                       ROOT / "benchmarks" / "indic_expanded_outputs.jsonl",
                       ROOT / "benchmarks" / "math_outputs.jsonl"],
        "provenance": [ROOT / "adivyanta_indic" / "data_manifest.json",
                       ROOT / "adivyanta_indic" / "expanded_data_manifest.json"],
    }.items():
        folder = PACKAGE / group
        folder.mkdir(exist_ok=True)
        for source in files:
            shutil.copy2(source, folder / source.name)
    print(f"Prepared {PACKAGE}")


if __name__ == "__main__":
    main()
