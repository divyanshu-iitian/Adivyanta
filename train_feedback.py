"""Optional local weight update from explicit /correct feedback with a validation gate."""
import argparse
import json
import os
from pathlib import Path
import random

import torch
from tokenizers import Tokenizer

from conversation_memory import DEFAULT_PATH, Memory
from model import GPT, GPTConfig
from train_indic import DATA, OUT, batch, evaluate, load_rows
from smart_chat import MODEL_CHOICES


def encode_corrections(memory, tok, block_size):
    eos = tok.token_to_id("<|eos|>")
    rows = []
    for question, answer in memory.data["corrections"].items():
        prefix = tok.encode("<|bos|><|user|>" + question + "<|assistant|>", add_special_tokens=False).ids
        response = tok.encode(answer, add_special_tokens=False).ids[:160] + [eos]
        prefix = prefix[-max(1, block_size - len(response)):]
        ids = prefix + response
        if len(ids) <= block_size and prefix:
            rows.append((ids[:-1], [-100] * (len(prefix) - 1) + response, "feedback"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DATA.parent / "personalized.pt")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Feedback weight training requires a CUDA GPU")
    memory = Memory(args.memory_file)
    if not memory.data["corrections"]:
        raise ValueError("No explicit corrections saved. Use /correct question => better answer in chat first.")
    device = torch.device("cuda")
    personalized = args.output
    base = personalized if personalized.exists() else next((path for path in MODEL_CHOICES if path.exists()), OUT / "best.pt")
    tokenizer_path = base.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = OUT / "tokenizer.json"
    tok = Tokenizer.from_file(str(tokenizer_path))
    state = torch.load(base, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**state["config"])).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if set(missing) - {"head.weight"} or unexpected:
        raise ValueError(f"Invalid checkpoint keys: {missing}, {unexpected}")
    corrections = encode_corrections(memory, tok, model.cfg.block_size)
    valid = load_rows(DATA / "valid.jsonl", tok, model.cfg.block_size)
    chat_valid = [row for row, raw in zip(valid, (json.loads(s) for s in (DATA / "valid.jsonl").read_text(encoding="utf-8").splitlines()))
                  if raw["source"] in {"hinglish", "oasst", "original"}]
    rng = random.Random(71)
    baseline_feedback = evaluate(model, corrections, device, limit=len(corrections))
    baseline_chat = evaluate(model, chat_valid, device, limit=len(chat_valid))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for _ in range(args.steps):
        chosen = [rng.randrange(len(corrections)) for _ in range(4)]
        x, y = batch(corrections, chosen, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    final_feedback = evaluate(model, corrections, device, limit=len(corrections))
    final_chat = evaluate(model, chat_valid, device, limit=len(chat_valid))
    accepted = final_feedback < baseline_feedback and final_chat <= baseline_chat + 0.10
    report = {"corrections": len(corrections), "steps": args.steps,
              "feedback_loss_before": round(baseline_feedback, 4), "feedback_loss_after": round(final_feedback, 4),
              "chat_validation_loss_before": round(baseline_chat, 4), "chat_validation_loss_after": round(final_chat, 4),
              "accepted": accepted}
    if accepted:
        weights = {name: tensor.detach().cpu().half() for name, tensor in model.state_dict().items()
                   if name != "head.weight"}
        temp = personalized.with_suffix(".tmp")
        torch.save({"model": weights, "config": model.config_dict(), "parameters": model.parameter_count(),
                    "local_feedback": len(corrections)}, temp)
        os.replace(temp, personalized)
    (DATA.parent / "feedback_training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
