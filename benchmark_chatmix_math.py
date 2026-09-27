"""Measure exact answers on generated arithmetic questions held out of ChatMix training."""
import argparse
import json
from pathlib import Path
import random
import re

import torch

from conversation_memory import Memory
from smart_chat import generate, load_model


ROOT = Path(__file__).parent


def number(text):
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data" / "chatmix_v2" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["source"] == "generated_math"]
    chosen = random.Random(20260928).sample(rows, min(args.limit, len(rows)))
    model, tok = load_model(args.checkpoint, prefer_local=False)
    memory = Memory(ROOT / "data" / "math_empty_memory.json")
    memory.data = {"facts": {}, "notes": [], "likes": [], "turns": [], "corrections": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with args.out.open("w", encoding="utf-8") as file:
        for index, row in enumerate(chosen, 1):
            torch.manual_seed(20260928 + index)
            response = generate(model, tok, memory, row["user"], max_new_tokens=15)
            gold, predicted = number(row["assistant"]), number(response)
            hit = gold == predicted and gold is not None
            correct += hit
            file.write(json.dumps({"id": index, "question": row["user"], "gold": gold,
                                   "response": response, "predicted": predicted, "correct": hit}, ensure_ascii=False) + "\n")
    report = {"checkpoint": str(args.checkpoint), "dataset": "ChatMix v2 generated_math test",
              "questions": len(chosen), "exact_answers": correct,
              "accuracy": round(correct / len(chosen), 4),
              "caveat": "Short generated arithmetic only; this does not measure word-problem reasoning or general intelligence."}
    summary = args.out.with_suffix(".summary.json")
    summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
