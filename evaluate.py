"""Check held-out loss and print a few reproducible example replies."""
import math
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from chat import respond
from model import GPT, GPTConfig
from train import evaluate, load_examples


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
ckpt = torch.load(ROOT / "checkpoints" / "best.pt", map_location=device, weights_only=False)
model = GPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
test = load_examples(DATA / "test.jsonl", tok, model.cfg.block_size)
loss = evaluate(model, test, device, torch.float16, n=len(test), weighted=True)
(ROOT / "checkpoints" / "evaluation.json").write_text(json.dumps({"checkpoint_step": ckpt["step"], "test_examples": len(test), "response_token_loss": round(loss, 4), "perplexity": round(math.exp(loss), 2)}, indent=2), encoding="utf-8")
print(f"checkpoint_step={ckpt['step']} test_examples={len(test)} test_loss={loss:.4f} perplexity={math.exp(loss):.2f}")
for prompt in ("I feel lonely today.", "I got a new job!", "My friend forgot my birthday.", "What are you?"):
    torch.manual_seed(123)
    print(f"You: {prompt}\nSmallGPT: {respond(model, tok, [prompt])}\n")
