"""Generate transparent, fixed-prompt comparisons with both source models."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat import respond as tiny_respond
from chat_adivyanta import respond as large_respond
from fine_tune_360 import BASE, BASE_REVISION
from model import GPT, GPTConfig


ROOT = Path(__file__).parent


def load_tiny(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [json.loads(s) for s in (ROOT / "data" / "roast" / "benchmark_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        prompts = prompts[:args.limit]
    tok_path = ROOT / "data" / "processed" / "tokenizer.json"
    if not tok_path.exists():
        tok_path = ROOT / "adivyanta_10m" / "tokenizer.json"
    tok_tiny = Tokenizer.from_file(str(tok_path))
    tiny_base = load_tiny(ROOT / "checkpoints" / "best.pt", device)
    tiny_tuned = load_tiny(ROOT / "adivyanta_10m" / "best.pt", device)
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    dtype = torch.float16 if device == "cuda" else torch.float32
    large_base = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=dtype, attn_implementation="sdpa").to(device)
    large = PeftModel.from_pretrained(large_base, ROOT / "adivyanta_adapter")
    large.eval()
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    with (out / "raw_outputs.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(prompts):
            prompt = row["prompt"]
            torch.manual_seed(123)
            a = tiny_respond(tiny_base, tok_tiny, [prompt])
            torch.manual_seed(123)
            b = tiny_respond(tiny_tuned, tok_tiny, [prompt])
            history = [{"role": "user", "content": prompt}]
            with large.disable_adapter():
                c = large_respond(large, tok, history)
            d = large_respond(large, tok, history)
            result = {"id": i + 1, "kind": row["kind"], "prompt": prompt, "original_10m": a,
                      "adivyanta_10m": b, "base_360m": c, "adivyanta_360m": d}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
