"""LoRA fine-tune Adivyanta on original roast conversations."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).parent
BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
BASE_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"
SYSTEM = (
    "You are Adivyanta, a witty chat assistant. When the user explicitly asks for a roast, "
    "reply with one short, clever, playful line about the habit or hobby they volunteered. "
    "Avoid protected traits and serious hardship. When no roast is requested, reply normally and kindly."
)


def load_rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def encode_rows(rows, tok, max_len=192):
    examples = []
    for row in rows:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["user"]}],
            tokenize=True, add_generation_prompt=True,
        )
        answer = tok.encode(row["assistant"], add_special_tokens=False) + [tok.eos_token_id]
        if len(prompt) + len(answer) > max_len:
            prompt = prompt[-(max_len - len(answer)):]
        ids = prompt + answer
        labels = [-100] * len(prompt) + answer
        examples.append((ids, labels))
    return examples


def batch(examples, indices, pad_id, device):
    chosen = [examples[i] for i in indices]
    length = max(len(x) for x, _ in chosen)
    ids = torch.full((len(chosen), length), pad_id, dtype=torch.long)
    labels = torch.full((len(chosen), length), -100, dtype=torch.long)
    mask = torch.zeros((len(chosen), length), dtype=torch.long)
    for i, (x, y) in enumerate(chosen):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


@torch.no_grad()
def loss_on(model, examples, pad_id, device):
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(examples), 2):
        ids, labels, mask = batch(examples, range(start, min(start + 2, len(examples))), pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        n = (labels != -100).sum().item()
        total += loss.item() * n
        tokens += n
    model.train()
    return total / tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This fine-tune requires a CUDA GPU")
    torch.manual_seed(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=torch.float16, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
                      target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    train = encode_rows(load_rows(ROOT / "data" / "roast" / "train.jsonl"), tok)
    valid = encode_rows(load_rows(ROOT / "data" / "roast" / "valid.jsonl"), tok)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    out = ROOT / "adivyanta_adapter"
    out.mkdir(exist_ok=True)
    baseline = loss_on(model, valid, tok.pad_token_id, device)
    best = baseline
    report = {"base_model": BASE, "base_revision": BASE_REVISION, "train_examples": len(train), "valid_examples": len(valid),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "base_validation_loss": round(baseline, 4), "epochs": []}
    print(json.dumps({k: v for k, v in report.items() if k != "epochs"}), flush=True)
    for epoch in range(1, args.epochs + 1):
        indices = list(range(len(train)))
        random.shuffle(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for i in range(0, len(indices), 2):
            chosen = indices[i:i + 2]
            ids, labels, mask = batch(train, chosen, tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            running += loss.item()
            scaler.scale(loss / 4).backward()
            if (i // 2 + 1) % 4 == 0 or i + 2 >= len(indices):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        val = loss_on(model, valid, tok.pad_token_id, device)
        report["epochs"].append({"epoch": epoch, "train_batch_loss": round(running / math.ceil(len(indices) / 2), 4), "validation_loss": round(val, 4)})
        print(json.dumps(report["epochs"][-1]), flush=True)
        if val < best:
            best = val
            model.save_pretrained(out, safe_serialization=True)
            tok.save_pretrained(out)
            report["best_epoch"] = epoch
            report["best_validation_loss"] = round(best, 4)
        (out / "training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"best_epoch={report.get('best_epoch')} best_validation_loss={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
