"""Train a 46M parameter multilingual GPT from random weights; no pretrained model."""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "indic"
OUT = ROOT / "adivyanta_indic"


def load_rows(path, tok, block_size):
    eos = tok.token_to_id("<|eos|>")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prefix = tok.encode("<|bos|><|user|>" + row["user"] + "<|assistant|>", add_special_tokens=False).ids
        response = tok.encode(row["assistant"], add_special_tokens=False).ids[:160] + [eos]
        prefix = prefix[-max(1, block_size - len(response)):]
        ids = prefix + response
        if len(ids) > block_size:
            ids = ids[-block_size:]
            prefix = prefix[-(block_size - len(response)):]
        if not prefix or len(ids) < 3:
            continue
        rows.append((ids[:-1], [-100] * (len(prefix) - 1) + response, row["language"]))
    return rows


def batch(rows, indices, device):
    selected = [rows[i] for i in indices]
    length = max(len(row[0]) for row in selected)
    x = torch.zeros((len(selected), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (ids, labels, _) in enumerate(selected):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(labels)] = torch.tensor(labels)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, rows, device, limit=600):
    model.eval()
    rng = random.Random(31415)
    indices = rng.sample(range(len(rows)), min(limit, len(rows)))
    sums, counts = defaultdict(float), defaultdict(int)
    for start in range(0, len(indices), 8):
        ix = indices[start:start + 8]
        x, y = batch(rows, ix, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        tokens = (y != -100).sum().item()
        sums["all"] += loss.item() * tokens
        counts["all"] += tokens
    model.train()
    return sums["all"] / counts["all"]


def save_best(model, step, loss, out, tokenizer_path):
    out.mkdir(parents=True, exist_ok=True)
    # head.weight shares token.weight in GPT; omit the duplicate in the public checkpoint.
    state = {name: tensor.detach().cpu().half() for name, tensor in model.state_dict().items()
             if name != "head.weight"}
    torch.save({"model": state, "config": model.config_dict(), "step": step,
                "val_loss": loss, "parameters": model.parameter_count()}, out / "best.pt")
    (out / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("Choose --resume or --init-checkpoint, not both")
    if not torch.cuda.is_available():
        raise RuntimeError("Training this model requires a CUDA GPU; inference also works on CPU.")
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    data_dir, out_dir = args.data_dir, args.out_dir
    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = OUT / "tokenizer.json"
    tok = Tokenizer.from_file(str(tokenizer_path))
    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), block_size=256, n_layer=12, n_head=8, n_embd=512)
    model = GPT(cfg).to(device)
    train = load_rows(data_dir / "train.jsonl", tok, cfg.block_size)
    valid = load_rows(data_dir / "valid.jsonl", tok, cfg.block_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    start, best = 0, float("inf")
    if args.resume:
        state = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best = state["step"], state["best_val"]
    elif args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if set(missing) - {"head.weight"} or unexpected:
            raise ValueError(f"Invalid initialization checkpoint: {missing}, {unexpected}")
        start = state["step"]
        best = evaluate(model, valid, device)
        save_best(model, start, best, out_dir, tokenizer_path)
    report = {"parameters": model.parameter_count(), "train_examples": len(train),
              "valid_examples": len(valid), "device": str(device), "data_dir": str(data_dir),
              "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
              "initial_validation_loss": round(best, 4) if args.init_checkpoint else None, "steps": []}
    if (out_dir / "training_metrics.json").exists() and args.resume:
        report = json.loads((out_dir / "training_metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"parameters": model.parameter_count(), "train_examples": len(train),
                      "valid_examples": len(valid), "start_step": start}), flush=True)
    rng = random.Random(43 + start)
    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        scale = min(1.0, step / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.accum):
            ix = [rng.randrange(len(train)) for _ in range(args.batch_size)]
            x, y = batch(train, ix, device)
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = model(x, y)
            scaler.scale(loss / args.accum).backward()
            train_loss += loss.item() / args.accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0:
            print(f"step={step} train_loss={train_loss:.4f} elapsed_s={time.time()-t0:.0f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, valid, device)
            record = {"step": step, "train_loss": round(train_loss, 4), "validation_loss": round(val, 4),
                      "elapsed_seconds": round(time.time() - t0)}
            report["steps"].append(record)
            print(json.dumps(record), flush=True)
            if val < best:
                best = val
                save_best(model, step, val, out_dir, tokenizer_path)
                report["best_step"] = step
                report["best_validation_loss"] = round(val, 4)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "best_val": best}, out_dir / "last.pt")
            (out_dir / "training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
