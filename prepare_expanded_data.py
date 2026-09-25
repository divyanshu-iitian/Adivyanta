"""Expand the scratch model's training corpus without touching its held-out prompts."""
import json
import os
from pathlib import Path
import random

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from prepare_indic_data import OUT as ORIGINAL, clean, source_file, split_for, valid_pair


ROOT = Path(__file__).parent
OUT = ROOT / "data" / "indic_expanded"
ULTRA = ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a",
         "data/train_sft-00000-of-00003-a3ecf92756993583.parquet")
GSM = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
       "main/train-00000-of-00001.parquet")


def download(source):
    repo, revision, filename = source
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def normalize(prompt):
    return clean(prompt).casefold()


def extra_hindi(existing, limit=100000):
    rng = random.Random(2026)
    kept, seen = [], 0
    for batch in pq.ParquetFile(source_file("aya_hindi")).iter_batches(batch_size=2048,
                              columns=["inputs", "targets", "task_type"]):
        for row in batch.to_pylist():
            if row["task_type"] not in {"question-answering", "generation"}:
                continue
            prompt, answer = clean(row["inputs"]), clean(row["targets"])
            if not valid_pair(prompt, answer) or normalize(prompt) in existing:
                continue
            seen += 1
            item = {"user": prompt, "assistant": answer, "language": "hin", "source": "aya_hindi_extra"}
            if len(kept) < limit:
                kept.append(item)
            else:
                j = rng.randrange(seen)
                if j < limit:
                    kept[j] = item
    return kept, seen


def ultrachat_pairs(path):
    result = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["messages"]):
        for row in batch.to_pylist():
            messages = row["messages"] or []
            for i in range(1, len(messages)):
                if messages[i]["role"] != "assistant" or messages[i - 1]["role"] != "user":
                    continue
                prompt, answer = clean(messages[i - 1]["content"]), clean(messages[i]["content"])
                if valid_pair(prompt, answer):
                    result.append({"user": prompt, "assistant": answer, "language": "eng", "source": "ultrachat"})
    return result


def gsm_pairs(path):
    result = []
    for row in pq.read_table(path, columns=["question", "answer"]).to_pylist():
        prompt = clean(row["question"])
        raw = row["answer"] or ""
        if "####" not in raw:
            continue
        reasoning, final = raw.rsplit("####", 1)
        answer = clean(reasoning[:320]) + " Therefore, the answer is " + clean(final) + "."
        if valid_pair(prompt, answer):
            result.append({"user": prompt, "assistant": answer, "language": "eng", "source": "gsm8k"})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    buckets = {name: [json.loads(s) for s in (ORIGINAL / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
               for name in ("train", "valid", "test")}
    existing = {normalize(row["user"]) for rows in buckets.values() for row in rows}
    hindi, hindi_seen = extra_hindi(existing)
    ultra = ultrachat_pairs(download(ULTRA))
    gsm = gsm_pairs(download(GSM))
    added = {"aya_hindi_extra": 0, "ultrachat": 0, "gsm8k": 0}
    for row in hindi + ultra + gsm:
        key = normalize(row["user"])
        if key in existing:
            continue
        existing.add(key)
        buckets[split_for(row["user"])].append(row)
        added[row["source"]] += 1
    rng = random.Random(2026)
    for name, rows in buckets.items():
        rng.shuffle(rows)
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"original_splits": {name: len((ORIGINAL / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()) for name in buckets},
              "eligible_extra_hindi": hindi_seen, "added": added,
              "expanded_splits": {name: len(rows) for name, rows in buckets.items()},
              "sources": {"ultrachat": {"repo": ULTRA[0], "revision": ULTRA[1], "license": "MIT"},
                          "gsm8k": {"repo": GSM[0], "revision": GSM[1], "license": "MIT"},
                          "aya_hindi_extra": {"repo": "CohereLabs/aya_collection_language_split", "license": "Apache-2.0"}},
              "tokenizer": "Existing 16,384-token Adivyanta Indic tokenizer; no pretrained tokenizer."}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
