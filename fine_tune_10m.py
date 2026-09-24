"""Continue the original scratch-trained 9.7M model on roast replies."""
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from train import batch, evaluate, load_examples


ROOT = Path(__file__).parent
ROAST = ROOT / "data" / "roast"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "adivyanta_10m"


def roast_examples(path, tok, block_size):
    examples = []
    eos = tok.token_to_id("<|eos|>")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prefix = tok.encode("<|bos|><|user|>" + row["user"] + "<|assistant|>", add_special_tokens=False).ids
        answer = tok.encode(row["assistant"], add_special_tokens=False).ids[:80] + [eos]
        prefix = prefix[-(block_size - len(answer)):]
        ids = prefix + answer
        examples.append((ids[:-1], [-100] * (len(prefix) - 1) + answer))
    return examples


def main():
    random.seed(45)
    torch.manual_seed(45)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(str(PROCESSED / "tokenizer.json"))
    ckpt = torch.load(ROOT / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    roast_train = roast_examples(ROAST / "train.jsonl", tok, model.cfg.block_size)
    roast_valid = roast_examples(ROAST / "valid.jsonl", tok, model.cfg.block_size)
    general = load_examples(PROCESSED / "train.jsonl", tok, model.cfg.block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    OUT.mkdir(exist_ok=True)
    baseline = evaluate(model, roast_valid, device, torch.float16, n=len(roast_valid), weighted=True)
    best = baseline
    report = {"base_checkpoint_step": ckpt["step"], "train_examples": len(roast_train), "valid_examples": len(roast_valid),
              "base_roast_validation_loss": round(baseline, 4), "evaluations": []}
    print(json.dumps({k: v for k, v in report.items() if k != "evaluations"}), flush=True)
    for step in range(1, 601):
        pool = roast_train if random.random() < 0.8 else general
        indices = [random.randrange(len(pool)) for _ in range(16)]
        x, y = batch(pool, indices, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 50 == 0:
            val = evaluate(model, roast_valid, device, torch.float16, n=len(roast_valid), weighted=True)
            record = {"step": step, "train_batch_loss": round(loss.item(), 4), "roast_validation_loss": round(val, 4)}
            report["evaluations"].append(record)
            print(json.dumps(record), flush=True)
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "config": model.config_dict(), "step": step,
                            "val_loss": val, "parameters": model.parameter_count()}, OUT / "best.pt")
                (OUT / "tokenizer.json").write_bytes((PROCESSED / "tokenizer.json").read_bytes())
                report["best_step"] = step
                report["best_roast_validation_loss"] = round(val, 4)
            (OUT / "training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
