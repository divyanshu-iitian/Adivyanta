"""Train the 9.7M parameter GPT from random weights on chat responses."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "checkpoints"


def load_examples(path, tok, block_size):
    examples = []
    eos = tok.token_to_id("<|eos|>")
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prefix = tok.encode(row["prefix"], add_special_tokens=False).ids
            response = tok.encode(row["response"], add_special_tokens=False).ids[:80] + [eos]
            prefix = prefix[-(block_size - len(response)) :]
            ids = prefix + response
            if len(ids) < 3 or len(prefix) < 1:
                continue
            x = ids[:-1]
            y = [-100] * (len(prefix) - 1) + response
            examples.append((x, y))
    return examples


def batch(examples, indices, device):
    selected = [examples[i] for i in indices]
    length = max(len(x) for x, _ in selected)
    x = torch.zeros((len(selected), length), dtype=torch.long)
    y = torch.full((len(selected), length), -100, dtype=torch.long)
    for i, (tokens, labels) in enumerate(selected):
        x[i, :len(tokens)] = torch.tensor(tokens)
        y[i, :len(labels)] = torch.tensor(labels)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, examples, device, amp_dtype, n=64, weighted=False):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    indices = random.Random(31415).sample(range(len(examples)), min(n, len(examples)))
    for start in range(0, len(indices), 8):
        x, y = batch(examples, indices[start:start + 8], device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        count = (y != -100).sum().item() if weighted else 1
        loss_sum += loss.item() * count
        token_count += count
    model.train()
    return loss_sum / token_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    cfg = GPTConfig(vocab_size=tok.get_vocab_size())
    model = GPT(cfg).to(device)
    train = load_examples(DATA / "train.jsonl", tok, cfg.block_size)
    valid = load_examples(DATA / "valid.jsonl", tok, cfg.block_size)
    OUT.mkdir(exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_step = 0
    best_val = float("inf")
    if args.resume and (OUT / "last.pt").exists():
        ckpt = torch.load(OUT / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        if device.type == "cuda":
            for group in optimizer.param_groups:
                group["fused"] = True
        start_step = ckpt["step"]
        best_val = ckpt["best_val"]
    print(json.dumps({"parameters": model.parameter_count(), "device": str(device), "train_examples": len(train), "valid_examples": len(valid), "starting_step": start_step}), flush=True)
    rng = random.Random(43)
    t0 = time.time()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        lr_scale = min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale
        indices = [rng.randrange(len(train)) for _ in range(args.batch_size)]
        x, y = batch(train, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 50 == 0:
            print(f"step={step} train_loss={loss.item():.4f} elapsed_s={time.time()-t0:.0f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, valid, device, torch.float16)
            print(f"step={step} val_loss={val_loss:.4f} elapsed_s={time.time()-t0:.0f}", flush=True)
            state = {"model": model.state_dict(), "config": model.config_dict(), "step": step, "val_loss": val_loss, "parameters": model.parameter_count()}
            if val_loss < best_val:
                best_val = val_loss
                torch.save(state, OUT / "best.pt")
            torch.save({**state, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "best_val": best_val}, OUT / "last.pt")
            (OUT / "metrics.json").write_text(json.dumps({"step": step, "val_loss": val_loss, "best_val_loss": best_val, "elapsed_seconds": round(time.time() - t0), "parameters": model.parameter_count(), "device": str(device)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
