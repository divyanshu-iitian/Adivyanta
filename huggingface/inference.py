"""Run either Adivyanta safetensors checkpoint from a downloaded model folder."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def generate(prompt, variant="base", seed=42, max_new_tokens=80):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(**json.loads((ROOT / "config.json").read_text(encoding="utf-8")))
    model = GPT(config).to(device)
    filename = "chat_model.safetensors" if variant == "chat" else "model.safetensors"
    missing, unexpected = model.load_state_dict(load_file(str(ROOT / filename), device=device), strict=False)
    if set(missing) != {"head.weight"} or unexpected:
        raise ValueError(f"Invalid checkpoint: missing={missing}, unexpected={unexpected}")
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
    ids = tok.encode("<|bos|><|user|>" + prompt + "<|assistant|>", add_special_tokens=False).ids
    ids = ids[-(config.block_size - max_new_tokens):]
    prompt_len = len(ids)
    stop = {tok.token_to_id(token) for token in ("<|eos|>", "<|user|>", "<|assistant|>", "<|pad|>")}
    for _ in range(max_new_tokens):
        logits, _ = model(torch.tensor([ids], device=device))
        scores = logits[0, -1].float() / 0.75
        scores[tok.token_to_id("<|pad|>")] = -float("inf")
        threshold = torch.topk(scores, min(30, scores.numel())).values[-1]
        scores[scores < threshold] = -float("inf")
        next_id = torch.multinomial(torch.softmax(scores, dim=-1), 1).item()
        if next_id in stop:
            break
        ids.append(next_id)
    return tok.decode(ids[prompt_len:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--variant", choices=("base", "chat"), default="base")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate(args.prompt, args.variant, args.seed))
