"""Chat with Adivyanta's stronger, LoRA-adapted 360M model."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tune_360 import BASE, BASE_REVISION, SYSTEM
from curated_roast import choose


ROOT = Path(__file__).parent
NORMAL_SYSTEM = "You are a helpful, respectful assistant. Answer normally and kindly. Do not make jokes about someone's distress or identity."
ROAST_RE = re.compile(r"\b(roast\w*|make fun|joke about|mazaak|mazak|leg.pull|savage|playful line)\b", re.I)
NO_ROAST_RE = re.compile(r"\b(?:don't|do not|no)\s+(?:(?:want|need)\s+)?(?:a\s+)?roast\b", re.I)
SENSITIVE_RE = re.compile(r"\b(religion|religious|caste|race|ethnicity|disability|illness|disease|death)\b", re.I)


def load_model(base_only=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=dtype, attn_implementation="sdpa").to(device)
    if not base_only:
        model = PeftModel.from_pretrained(model, ROOT / "adivyanta_adapter")
    model.eval()
    return model, tok


@torch.inference_mode()
def respond(model, tok, history, sample=False, system=SYSTEM):
    messages = [{"role": "system", "content": system}] + history[-6:]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    kwargs = {"max_new_tokens": 64, "do_sample": sample, "pad_token_id": tok.eos_token_id}
    if sample:
        kwargs.update(temperature=0.75, top_p=0.9)
    output = model.generate(ids, attention_mask=torch.ones_like(ids), **kwargs)
    return tok.decode(output[0, ids.shape[-1]:], skip_special_tokens=True).strip()


def route_name(prompt, curated=True):
    if NO_ROAST_RE.search(prompt):
        return "base"
    if ROAST_RE.search(prompt):
        if SENSITIVE_RE.search(prompt):
            return "redirect"
        if curated and choose(prompt)[0]:
            return "curated"
        return "adapter"
    return "base"


def respond_routed(model, tok, history, sample=False, curated=True):
    """Default chat: curated covered roasts, adapted model elsewhere, base for normal chat."""
    route = route_name(history[-1]["content"], curated)
    if route == "redirect":
        return "Let's keep the roast about a habit or hobby you chose. Give me one and I'll make it playful."
    if route == "curated":
        return choose(history[-1]["content"])[0]
    if route == "adapter":
        return respond(model, tok, history, sample=sample)
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with context:
        return respond(model, tok, history, sample=sample, system=NORMAL_SYSTEM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt")
    parser.add_argument("--base-only", action="store_true", help="Run the unfine-tuned SmolLM2 baseline")
    parser.add_argument("--model-only", action="store_true", help="Run the raw adapted model without chat routing")
    parser.add_argument("--no-curated", action="store_true", help="Skip authored roast lines for recognized topics")
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    args = parser.parse_args()
    model, tok = load_model(args.base_only)
    answer_fn = respond if args.model_only or args.base_only else lambda m, t, h, s: respond_routed(m, t, h, s, curated=not args.no_curated)
    if args.prompt:
        print(answer_fn(model, tok, [{"role": "user", "content": args.prompt}], args.sample))
        return
    print("Adivyanta chat. Playful roasts on request; type /quit to exit.")
    history = []
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == "/quit":
            break
        if not prompt:
            continue
        history.append({"role": "user", "content": prompt})
        answer = answer_fn(model, tok, history, args.sample)
        print("Adivyanta:", answer)
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
