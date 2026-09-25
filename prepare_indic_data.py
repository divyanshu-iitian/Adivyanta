"""Build a permissively licensed English/Hindi/Hinglish chat corpus and tokenizer."""
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).parent
OUT = ROOT / "data" / "indic"
SPECIAL = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|unk|>"]
SOURCES = {
    "aya_human": ("CohereLabs/aya_collection", "09069079fad96ad9c7f8781be8c1d70471dfa768", "aya_dataset/train-00000-of-00001.parquet"),
    "aya_hindi": ("CohereLabs/aya_collection_language_split", "a3af2fde4b4cb5b2775830b11244a1a20b5f004f", "hindi/train-00000-of-00003.parquet"),
    "aya_english": ("CohereLabs/aya_collection_language_split", "a3af2fde4b4cb5b2775830b11244a1a20b5f004f", "english/train-00001-of-00007.parquet"),
    "oasst": ("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b", "data/train-00000-of-00001-b42a775f407cee45.parquet"),
    "hinglish": ("DSMJ910/hinglish-instruct-10k", "39686808306eb6d898c8f7e86dfa93549e4ef23b", "data/train-00000-of-00001.parquet"),
}
INDIC_LANGS = {"hin", "eng", "ben", "mar", "tam", "tel", "guj", "kan", "mal", "pan", "urd"}


def source_file(name):
    repo, revision, filename = SOURCES[name]
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def clean(text):
    return " ".join(str(text).split()).strip()


def valid_pair(prompt, answer):
    return (4 <= len(prompt) <= 450 and 2 <= len(answer) <= 600
            and "�" not in prompt + answer and "<unk>" not in prompt + answer)


def collect_aya(path, limit, allowed_tasks=None, allowed_langs=None, seed=42):
    """Reservoir sampling avoids taking only the first source in a sorted shard."""
    rng = random.Random(seed)
    kept, seen = [], 0
    columns = ["inputs", "targets", "task_type", "language"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            if allowed_tasks and row["task_type"] not in allowed_tasks:
                continue
            if allowed_langs and row["language"] not in allowed_langs:
                continue
            prompt, answer = clean(row["inputs"]), clean(row["targets"])
            if not valid_pair(prompt, answer):
                continue
            seen += 1
            item = {"user": prompt, "assistant": answer, "language": row["language"]}
            if len(kept) < limit:
                kept.append(item)
            else:
                i = rng.randrange(seen)
                if i < limit:
                    kept[i] = item
    return kept, seen


def collect_oasst(path):
    rows = pq.read_table(path, columns=["message_id", "parent_id", "text", "role", "lang", "deleted", "review_result", "labels"]).to_pylist()
    by_id = {row["message_id"]: row for row in rows}
    result = []
    for row in rows:
        if row["role"] != "assistant" or row["lang"] not in {"en", "hi"} or row["deleted"] or row["review_result"] is False:
            continue
        parent = by_id.get(row["parent_id"])
        if not parent or parent["role"] != "prompter" or parent["deleted"]:
            continue
        prompt, answer = clean(parent["text"]), clean(row["text"])
        if valid_pair(prompt, answer):
            result.append({"user": prompt, "assistant": answer, "language": "eng" if row["lang"] == "en" else "hin"})
    return result


def collect_hinglish(path):
    rows = pq.read_table(path, columns=["user_message", "assistant_response", "category"]).to_pylist()
    result = []
    for row in rows:
        if row["category"] not in {"casual", "question"}:
            continue
        prompt, answer = clean(row["user_message"]), clean(row["assistant_response"])
        if valid_pair(prompt, answer):
            result.append({"user": prompt, "assistant": answer, "language": "hinglish"})
    return result


def hand_written():
    """Small original examples for greetings and short conversational turns."""
    return [
        ("hi", "Hi! Kya haal hai?"),
        ("hello", "Hello! Kaise ho?"),
        ("kya kar rahe ho", "Tumse baat kar raha hoon. Batao, kya chal raha hai?"),
        ("kaise ho", "Main theek hoon. Tum kaise ho?"),
        ("samosa", "Samosa! Chai ke saath best lagta hai. Tumhe teekha pasand hai?"),
        ("aaj kya karein", "Kuch naya try karein? Batao, mood kaisa hai?"),
        ("mere exam aa rahe hain", "Chalo ek chhota study plan banate hain. Kaunsa subject tough lag raha hai?"),
        ("mera code nahi chal raha", "Error message bhejo, saath mein debug karte hain."),
        ("aaj mood off hai", "Kya hua? Agar baat karna chaho, main sun raha hoon."),
        ("tell me a joke", "Mera calendar mujhse zyada busy hai, par kaam phir bhi pending hai."),
        ("can we chat in Hindi", "Haan, bilkul. Hindi mein baat kar sakte hain."),
        ("तुम क्या कर रहे हो", "मैं तुमसे बात कर रहा हूँ। बताओ, क्या हाल है?"),
        ("समोसा", "समोसा और चाय बढ़िया जोड़ी है। तुम्हें कैसा पसंद है?"),
        ("नमस्ते", "नमस्ते! आप कैसे हैं?"),
        ("I am learning Python", "Nice! What are you building with Python?"),
        ("I feel lonely", "I'm sorry you're feeling lonely. Want to tell me what has been going on?"),
    ]


def split_for(prompt):
    number = int(hashlib.sha256(prompt.casefold().encode("utf-8")).hexdigest()[:8], 16) % 100
    return "valid" if number == 0 else "test" if number == 1 else "train"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    human, human_seen = collect_aya(source_file("aya_human"), 35000, allowed_langs=INDIC_LANGS)
    hindi, hindi_seen = collect_aya(source_file("aya_hindi"), 50000,
                                     allowed_tasks={"question-answering", "generation"}, allowed_langs={"hin"})
    english, english_seen = collect_aya(source_file("aya_english"), 50000, allowed_langs={"eng"})
    oasst = collect_oasst(source_file("oasst"))
    hinglish = collect_hinglish(source_file("hinglish"))
    original = [{"user": user, "assistant": answer, "language": "hinglish" if user.isascii() else "hin"}
                for user, answer in hand_written()]
    sources = {"aya_human": human, "aya_hindi": hindi, "aya_english": english,
               "oasst": oasst, "hinglish": hinglish, "original": original}
    buckets = {"train": [], "valid": [], "test": []}
    seen_prompts = set()
    for name, rows in sources.items():
        for row in rows:
            key = clean(row["user"]).casefold()
            if key in seen_prompts:
                continue
            seen_prompts.add(key)
            row["source"] = name
            buckets[split_for(row["user"])].append(row)
    rng = random.Random(42)
    for rows in buckets.values():
        rng.shuffle(rows)
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    texts = (text for row in buckets["train"] for text in (row["user"], row["assistant"]))
    tok.train_from_iterator(texts, trainer=trainers.BpeTrainer(vocab_size=16384, special_tokens=SPECIAL,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
                            length=2 * len(buckets["train"]))
    tok.save(str(OUT / "tokenizer.json"))
    for split, rows in buckets.items():
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"sources": {name: len(rows) for name, rows in sources.items()},
              "eligible_aya_rows": {"human": human_seen, "hindi": hindi_seen, "english": english_seen},
              "splits": {name: len(rows) for name, rows in buckets.items()}, "vocab_size": tok.get_vocab_size(),
              "licenses": {"aya_collection": "Apache-2.0", "oasst1": "Apache-2.0", "hinglish_instruct_10k": "Apache-2.0", "original": "original"},
              "dataset_revisions": {name: revision for name, (_, revision, _) in SOURCES.items()}}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
