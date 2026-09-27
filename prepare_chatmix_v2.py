"""Build a licensed, filtered chat corpus with deterministic source-level provenance."""
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import random
import re

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from prepare_indic_data import clean, split_for


ROOT = Path(__file__).parent
SOURCE = ROOT / "data" / "indic_expanded"
OUT = ROOT / "data" / "chatmix_v2"
TOK = ROOT / "adivyanta_indic" / "expanded" / "tokenizer.json"
OASST2_REPO = "OpenAssistant/oasst2"
OASST2_REV = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
OASST2_FILE = "data/train-00000-of-00001-88ba0162028a73fc.parquet"
LICENSES = {
    "aya_human": "Apache-2.0", "aya_hindi": "Apache-2.0", "aya_hindi_extra": "Apache-2.0",
    "aya_english": "Apache-2.0", "oasst": "Apache-2.0", "oasst2": "Apache-2.0",
    "hinglish": "Apache-2.0", "ultrachat": "MIT", "gsm8k": "MIT", "original": "MIT",
    "generated_math": "MIT",
}
QUOTAS = {
    "aya_human": 20000, "aya_hindi": 15000, "aya_hindi_extra": 25000,
    "aya_english": 12000, "oasst": 6000, "oasst2": 14000,
    "hinglish": 6000, "ultrachat": 20000, "gsm8k": 7000,
    "original": 100, "generated_math": 10000,
}
TEMPLATE = re.compile(
    r"\b(?:PersonX|PersonY|PersonZ|translate|translation|summari[sz]e|given the following|"
    r"given passage|based on the passage|given the passage|"
    r"write a short paragraph of narrative|fill in the blank|choose the correct answer)\b"
    r"|(?:तथ्यों को मिलाकर|विस्तृत प्रश्न|विचार श्रृंखला|इस विचार श्रृंखला)", re.I
)
GENERIC = re.compile(
    r"^(?:I'm sorry|I am sorry|As an AI|As a language model|I cannot fulfill|I can't assist|"
    r"Sure[,!. ]|Certainly[,!. ])\s*$", re.I
)
PRIVATE = re.compile(r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://\S+|\b\+?\d[\d ()-]{8,}\d\b)")


def norm(text):
    return clean(text).casefold()


def quality(row, tok):
    user, answer = clean(row["user"]), clean(row["assistant"])
    min_answer_length = 1 if row["source"] == "generated_math" else 8
    if not (4 <= len(user) <= 350 and min_answer_length <= len(answer) <= 450):
        return False
    if TEMPLATE.search(user) or GENERIC.fullmatch(answer) or PRIVATE.search(user + " " + answer):
        return False
    if "\ufffd" in user + answer or "<|" in user + answer or answer.casefold() == user.casefold():
        return False
    # The model has 256 positions. Reject examples that would be heavily truncated.
    prefix = tok.encode("<|bos|><|user|>" + user + "<|assistant|>", add_special_tokens=False).ids
    target = tok.encode(answer, add_special_tokens=False).ids
    minimum = 1 if row["source"] == "generated_math" else 5
    return len(prefix) <= 140 and minimum <= len(target) <= 115 and len(prefix) + len(target) <= 240


def oasst2_pairs():
    path = hf_hub_download(OASST2_REPO, OASST2_FILE, repo_type="dataset", revision=OASST2_REV)
    rows = pq.read_table(path, columns=["message_id", "parent_id", "text", "role", "lang",
                                        "review_result", "deleted", "rank", "synthetic", "labels"]).to_pylist()
    parents = {row["message_id"]: row for row in rows}
    for row in rows:
        if row["role"] != "assistant" or row["lang"] not in {"en", "hi"}:
            continue
        if row["deleted"] or row["review_result"] is False or row["synthetic"]:
            continue
        if row["rank"] is not None and row["rank"] > 1:
            continue
        labels = row["labels"] or {}
        scores = dict(zip(labels.get("name") or [], labels.get("value") or []))
        if any(scores.get(label, 0) > 0.35 for label in ("spam", "pii", "toxicity", "hate_speech", "sexual_content")):
            continue
        if scores.get("quality", 1) < 0.4:
            continue
        parent = parents.get(row["parent_id"])
        if not parent or parent["role"] != "prompter" or parent["deleted"]:
            continue
        yield {"user": clean(parent["text"]), "assistant": clean(row["text"]),
               "language": "hin" if row["lang"] == "hi" else "eng", "source": "oasst2"}


def generated_math():
    rng = random.Random(20260928)
    seen = set()
    while len(seen) < 12000:
        a, b = rng.randrange(2, 300), rng.randrange(2, 300)
        op = rng.choice(["+", "-", "*"])
        if op == "*" and (a > 60 or b > 60):
            continue
        key = (a, b, op)
        if key in seen:
            continue
        seen.add(key)
        value = a + b if op == "+" else a - b if op == "-" else a * b
        if len(seen) % 3 == 0:
            prompt, answer, lang = f"{a} {op} {b} kitna hota hai?", f"{value}.", "hinglish"
        elif len(seen) % 3 == 1:
            prompt, answer, lang = f"Calculate {a} {op} {b}.", f"{value}.", "eng"
        else:
            prompt, answer, lang = f"{a} {op} {b} का उत्तर क्या है?", f"{value}।", "hin"
        yield {"user": prompt, "assistant": answer, "language": lang, "source": "generated_math"}


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def rank(row):
    text = row["source"] + "\0" + norm(row["user"])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(TOK))
    counts = Counter()
    candidates = defaultdict(list)
    for split in ("train", "valid", "test"):
        for row in read_jsonl(SOURCE / f"{split}.jsonl"):
            counts[(row["source"], "seen")] += 1
            if quality(row, tok):
                candidates[split].append(row)
                counts[(row["source"], "eligible")] += 1
    for row in oasst2_pairs():
        counts[("oasst2", "seen")] += 1
        if quality(row, tok):
            candidates[split_for(row["user"])].append(row)
            counts[("oasst2", "eligible")] += 1
    for row in generated_math():
        counts[("generated_math", "seen")] += 1
        if quality(row, tok):
            candidates[split_for(row["user"])].append(row)
            counts[("generated_math", "eligible")] += 1
    # Reserve held-out prompts before choosing any training examples.
    heldout = set()
    result = {name: [] for name in ("train", "valid", "test")}
    for split in ("valid", "test"):
        for row in sorted(candidates[split], key=rank):
            key = norm(row["user"])
            if key not in heldout:
                heldout.add(key)
                result[split].append(row)
    by_source = defaultdict(list)
    for row in candidates["train"]:
        if norm(row["user"]) not in heldout:
            by_source[row["source"]].append(row)
    selected = []
    for source, rows in sorted(by_source.items()):
        seen = set()
        for row in sorted(rows, key=rank):
            key = norm(row["user"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(seen) >= QUOTAS[source]:
                break
    # A prompt can occur under different sources; keep one deterministic answer.
    seen = set(heldout)
    for row in sorted(selected, key=rank):
        key = norm(row["user"])
        if key not in seen:
            seen.add(key)
            result["train"].append(row)
    rng = random.Random(20260928)
    for split, rows in result.items():
        rng.shuffle(rows)
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                row = {**row, "source_license": LICENSES[row["source"]],
                       "synthetic": row["source"] in {"hinglish", "ultrachat", "generated_math"}}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"version": "2.0.0", "selection": "Deterministic source quotas, prompt deduplication and length/quality filters",
              "source_revision_oasst2": OASST2_REV,
              "splits": {k: len(v) for k, v in result.items()},
              "train_by_source": dict(Counter(row["source"] for row in result["train"])),
              "train_by_language": dict(Counter(row["language"] for row in result["train"])),
              "candidate_counts": {source: {kind: counts[(source, kind)] for kind in ("seen", "eligible")}
                                   for source in LICENSES},
              "license_by_source": LICENSES,
              "previous_manifest": "adivyanta_indic/expanded_data_manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
