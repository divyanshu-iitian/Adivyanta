"""Select self-contained dialogue pairs from ChatMix v2 for a measured continuation."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).parent
SOURCE = ROOT / "data" / "chatmix_v2"
OUT = ROOT / "data" / "chat_focus"
QUOTAS = {"oasst2": 3500, "hinglish": 4000, "ultrachat": 4500,
          "aya_human": 3500, "aya_hindi": 2500, "aya_hindi_extra": 2500}
CONTEXT_DEPENDENT = re.compile(
    r"^(?:that|those|these|this|it|they|he|she|your|you|thanks|thank you|yes|no|sure|okay|ok|"
    r"as (?:you|we) (?:said|discussed)|(?:can|could) you (?:also|expand|elaborate))\b"
    r"|\b(?:as mentioned|from the (?:previous|above)|what you said|earlier answer|"
    r"that (?:sounds|makes sense|answered)|mentioned (?:earlier|above))\b", re.I
)
BOILERPLATE = re.compile(
    r"\b(?:if you have any (?:more|other) questions|feel free to ask|"
    r"don't hesitate to (?:ask|reach out)|as an ai language model)\b", re.I
)


def score(row):
    return hashlib.sha256((row["source"] + "\0" + row["user"]).encode()).hexdigest()


def accepted(row):
    user, answer = row["user"], row["assistant"]
    return (row["source"] in QUOTAS and 8 <= len(user) <= 220
            and 12 <= len(answer) <= 280
            and not CONTEXT_DEPENDENT.search(user)
            and not BOILERPLATE.search(answer))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "valid", "test"):
        rows = [json.loads(line) for line in (SOURCE / f"{split}.jsonl").open(encoding="utf-8")]
        selected = [row for row in rows if accepted(row)]
        if split == "train":
            by_source = Counter()
            chosen = []
            for row in sorted(selected, key=score):
                if by_source[row["source"]] < QUOTAS[row["source"]]:
                    chosen.append(row)
                    by_source[row["source"]] += 1
            selected = sorted(chosen, key=score)
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as target:
            for row in selected:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[split] = {"rows": len(selected), "sources": dict(Counter(row["source"] for row in selected))}
    (OUT / "manifest.json").write_text(json.dumps({
        "derived_from": "Adivyanta-ChatMix-v2", "selection": "deterministic self-contained dialogue filter",
        "quotas": QUOTAS, "splits": counts,
        "caveat": "Heuristic filtering; no claim of verified factual correctness or human quality."
    }, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
