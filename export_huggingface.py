"""Build a standalone Hugging Face model repo from local scratch checkpoints."""
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "adivyanta_indic" / "expanded"
REFINED = ROOT / "adivyanta_indic" / "chatmix_v2_refined"
PACKAGE = ROOT / "data" / "huggingface_upload"


def convert(source, checkpoint, output):
    state = torch.load(source / checkpoint, map_location="cpu", weights_only=False)
    if state["parameters"] != 46_349_312:
        raise ValueError("Unexpected parameter count")
    tensors = {key: value.contiguous() for key, value in state["model"].items() if key != "head.weight"}
    save_file(tensors, PACKAGE / output, metadata={"format": "pt", "source_checkpoint": checkpoint})
    return state["config"]


def main():
    PACKAGE.mkdir(parents=True, exist_ok=True)
    base = convert(SOURCE, "best.pt", "model.safetensors")
    chat = convert(SOURCE, "chat_best.pt", "chat_model.safetensors")
    refined = convert(REFINED, "best.pt", "chatmix_v2_model.safetensors")
    if base != chat or base != refined:
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
                    ROOT / "benchmarks" / "math_summary.json",
                    REFINED / "training_metrics.json", REFINED / "test_metrics.json",
                    REFINED / "legacy_test_metrics.json",
                    ROOT / "benchmarks" / "chatmix_math_v2.summary.json",
                    ROOT / "benchmarks" / "chatmix_v2_gsm8k_summary.json"],
        "benchmarks": [ROOT / "benchmarks" / "indic_expanded_base_outputs.jsonl",
                       ROOT / "benchmarks" / "indic_expanded_outputs.jsonl",
                       ROOT / "benchmarks" / "math_outputs.jsonl",
                       ROOT / "benchmarks" / "indic_chatmix_v2_outputs.jsonl",
                       ROOT / "benchmarks" / "indic_chatmix_v2_greedy_outputs.jsonl",
                       ROOT / "benchmarks" / "chatmix_math_v2.jsonl",
                       ROOT / "benchmarks" / "chatmix_v2_gsm8k_outputs.jsonl"],
        "provenance": [ROOT / "adivyanta_indic" / "data_manifest.json",
                       ROOT / "adivyanta_indic" / "expanded_data_manifest.json",
                       ROOT / "adivyanta_indic" / "chatmix_v2_manifest.json"],
    }.items():
        folder = PACKAGE / group
        folder.mkdir(exist_ok=True)
        for source in files:
            name = f"chatmix_v2_{source.name}" if source.parent == REFINED else source.name
            shutil.copy2(source, folder / name)
    print(f"Prepared {PACKAGE}")


if __name__ == "__main__":
    main()
