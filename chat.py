"""Adivyanta chat entry point; --scratch runs the educational 10M GPT."""
import argparse
from pathlib import Path
import sys

import torch

from model import GPT, GPTConfig


ROOT = Path(__file__).parent


@torch.inference_mode()
def respond(model, tok, history, max_new_tokens=64, temperature=0.6, top_k=20):
    prompt = "<|bos|>" + "".join(("<|user|>" if i % 2 == 0 else "<|assistant|>") + turn for i, turn in enumerate(history)) + "<|assistant|>"
    ids = tok.encode(prompt, add_special_tokens=False).ids[-(model.cfg.block_size - max_new_tokens):]
    prompt_len = len(ids)
    device = next(model.parameters()).device
    stop_ids = {tok.token_to_id(s) for s in ("<|eos|>", "<|user|>", "<|assistant|>", "<|pad|>")}
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.cfg.block_size:]], device=device)
        logits, _ = model(x)
        scores = logits[0, -1].float() / max(temperature, 0.1)
        scores[tok.token_to_id("<|pad|>")] = -float("inf")
        if top_k > 0:
            threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
            scores[scores < threshold] = -float("inf")
        next_id = torch.multinomial(torch.softmax(scores, dim=-1), 1).item()
        if next_id in stop_ids:
            break
        ids.append(next_id)
    return tok.decode(ids[prompt_len:], skip_special_tokens=True).strip()


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--prompt", help="Generate one response and exit")
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_path = Path(args.checkpoint).parent / "tokenizer.json"
    if not tok_path.exists():
        tok_path = ROOT / "data" / "processed" / "tokenizer.json"
    if not tok_path.exists():
        tok_path = ROOT / "adivyanta_10m" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    history = []
    if args.prompt:
        print(respond(model, tok, [args.prompt], temperature=args.temperature))
        return
    print("SmallGPT chat (English training data; not sentient). Type /quit to exit.")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if user == "/quit":
            break
        if not user:
            continue
        history.append(user)
        answer = respond(model, tok, history[-5:], temperature=args.temperature)
        print("SmallGPT:", answer)
        history.append(answer)


if __name__ == "__main__":
    if "--scratch" in sys.argv:
        if "--scratch" in sys.argv:
            sys.argv.remove("--scratch")
        main()
    else:
        from smart_chat import main as adivyanta_main

        adivyanta_main()
