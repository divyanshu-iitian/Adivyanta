"""Independent open-weight 0.5B baseline on the same fixed roast prompts."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tune_360 import SYSTEM


ROOT = Path(__file__).parent
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    rows = [json.loads(s) for s in (ROOT / "data" / "roast" / "benchmark_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    out = ROOT / "benchmarks"
    out.mkdir(exist_ok=True)
    with (out / "qwen_outputs.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}]
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                result = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False,
                                        pad_token_id=tok.eos_token_id)
            response = tok.decode(result[0, ids.shape[-1]:], skip_special_tokens=True).strip()
            record = {"id": i, "kind": row["kind"], "prompt": row["prompt"], "qwen_05b": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
    (out / "qwen_model.json").write_text(json.dumps({"model": BASE, "revision": BASE_REVISION}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
