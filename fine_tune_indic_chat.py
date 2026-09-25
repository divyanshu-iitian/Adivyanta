"""Continue Adivyanta's scratch weights on conversational examples only."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from train_indic import DATA, OUT, batch, evaluate, load_rows


def source_indices(path):
    return [json.loads(line)["source"] for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--base-checkpoint", type=Path, default=OUT / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Chat continuation requires a CUDA GPU")
    random.seed(55)
    torch.manual_seed(55)
    device = torch.device("cuda")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(args.base_checkpoint.parent / "tokenizer.json"))
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if set(missing) - {"head.weight"} or unexpected:
        raise ValueError(f"Invalid checkpoint keys: {missing}, {unexpected}")
    train = load_rows(args.data_dir / "train.jsonl", tok, model.cfg.block_size)
    valid = load_rows(args.data_dir / "valid.jsonl", tok, model.cfg.block_size)
    sources = source_indices(args.data_dir / "train.jsonl")
    valid_sources = source_indices(args.data_dir / "valid.jsonl")
    if len(train) != len(sources) or len(valid) != len(valid_sources):
        raise ValueError("Unexpected skipped row; source index alignment failed")
    pools = {name: [i for i, source in enumerate(sources) if source == name]
             for name in ("hinglish", "oasst", "aya_human", "original")}
    chat_valid = [row for row, source in zip(valid, valid_sources) if source in {"hinglish", "oasst", "original"}]
    baseline = evaluate(model, chat_valid, device, limit=len(chat_valid))
    best = baseline
    report = {"starting_checkpoint_step": ckpt["step"], "chat_validation_examples": len(chat_valid),
              "base_chat_validation_loss": round(baseline, 4), "train_pools": {k: len(v) for k, v in pools.items()}, "steps": []}
    print(json.dumps(report), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.02, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(55)
    stale_evaluations = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(2):
            pool_name = rng.choices(["hinglish", "oasst", "aya_human", "original"], weights=[39, 45, 15, 1])[0]
            pool = pools[pool_name]
            indices = [rng.choice(pool) for _ in range(8)]
            x, y = batch(train, indices, device)
            with torch.autocast("cuda", dtype=torch.float16):
                _, loss = model(x, y)
            scaler.scale(loss / 2).backward()
            total += loss.item() / 2
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 100 == 0:
            print(f"step={step} train_loss={total:.4f} elapsed_s={time.time()-t0:.0f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, chat_valid, device, limit=len(chat_valid))
            row = {"step": step, "chat_validation_loss": round(val, 4)}
            report["steps"].append(row)
            print(json.dumps(row), flush=True)
            if val < best:
                best = val
                stale_evaluations = 0
                state = {name: tensor.detach().cpu().half() for name, tensor in model.state_dict().items()
                         if name != "head.weight"}
                torch.save({"model": state, "config": model.config_dict(), "step": ckpt["step"],
                            "chat_steps": step, "val_loss": val, "parameters": model.parameter_count()}, out_dir / "chat_best.pt")
                (out_dir / "tokenizer.json").write_bytes((args.base_checkpoint.parent / "tokenizer.json").read_bytes())
                report["best_chat_step"] = step
                report["best_chat_validation_loss"] = round(val, 4)
            else:
                stale_evaluations += 1
            (out_dir / "chat_training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if stale_evaluations >= 3:
                print("Stopping after three evaluations without improvement", flush=True)
                break


if __name__ == "__main__":
    main()
