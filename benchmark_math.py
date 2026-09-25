"""Exact-answer check on held-out GSM8K test questions; sampled generation is seeded."""
import argparse
import json
import os
from pathlib import Path
import random
import re

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from conversation_memory import Memory
from smart_chat import generate, load_model


ROOT = Path(__file__).parent
REVISION = "740312add88f781978c0658806c59bc2815b9866"


def number(text):
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return matches[-1].replace(",", "") if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet",
                           repo_type="dataset", revision=REVISION)
    rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
    chosen = random.Random(2026).sample(rows, min(args.limit, len(rows)))
    model, tok = load_model(args.checkpoint, prefer_local=False)
    memory = Memory(ROOT / "data" / "math_benchmark_empty_memory.json")
    memory.data = {"facts": {}, "notes": [], "likes": [], "turns": [], "corrections": {}}
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    correct = 0
    with (out / "math_outputs.jsonl").open("w", encoding="utf-8") as file:
        for index, row in enumerate(chosen, 1):
            torch.manual_seed(2026 + index)
            response = generate(model, tok, memory, row["question"], max_new_tokens=100)
            gold = number(row["answer"].rsplit("####", 1)[-1])
            predicted = number(response)
            hit = gold == predicted and gold is not None
            correct += hit
            record = {"id": index, "question": row["question"], "gold": gold,
                      "response": response, "predicted": predicted, "correct": hit}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{index}/{len(chosen)} correct={correct}", flush=True)
    summary = {"dataset": "openai/gsm8k", "revision": REVISION, "split": "official test",
               "checkpoint": str(args.checkpoint) if args.checkpoint else "default released model",
               "questions": len(chosen), "exact_answers": correct, "accuracy": round(correct / len(chosen), 4),
               "caveat": "Small seeded subset. Last generated number is used as the answer; this is not a broad reasoning benchmark."}
    (out / "math_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
