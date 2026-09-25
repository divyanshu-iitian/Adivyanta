"""Evaluate the untouched multilingual test split and publish per-language losses."""
import json
import math
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from train_indic import DATA, OUT, batch, load_rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or (OUT / "chat_best.pt" if (OUT / "chat_best.pt").exists() else OUT / "best.pt")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    tokenizer_path = checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = OUT / "tokenizer.json"
    tok = Tokenizer.from_file(str(tokenizer_path))
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if set(missing) - {"head.weight"} or unexpected:
        raise ValueError(f"Invalid checkpoint keys: missing={missing}, unexpected={unexpected}")
    model.eval()
    rows = load_rows(args.data_dir / "test.jsonl", tok, model.cfg.block_size)
    groups = defaultdict(list)
    for row in rows:
        groups[row[2]].append(row)
    report = {"checkpoint": str(checkpoint), "checkpoint_step": ckpt.get("step"), "parameters": model.parameter_count(),
              "test_examples": len(rows), "by_language": {}}
    for language, examples in sorted(groups.items()):
        total, tokens = 0.0, 0
        for start in range(0, len(examples), 8):
            x, y = batch(examples, range(start, min(start + 8, len(examples))), device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            n = (y != -100).sum().item()
            total += loss.item() * n
            tokens += n
        mean = total / tokens
        report["by_language"][language] = {"examples": len(examples), "response_tokens": tokens,
                                           "loss": round(mean, 4), "perplexity": round(math.exp(mean), 2)}
    report["caveat"] = "These are test response-token losses. Languages with few examples have unstable estimates; losses do not measure chat quality."
    output = args.out or checkpoint.parent / "test_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
