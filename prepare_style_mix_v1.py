"""Local training mix that oversamples the public dialogue seed; not a dataset release."""
import hashlib
import json
from pathlib import Path
import shutil

from prepare_dialogue_seed_v1 import main as build_seed


ROOT = Path(__file__).parent
BASE = ROOT / "data" / "chatmix_v2"
SEED = ROOT / "data" / "dialogue_seed_v1"
OUT = ROOT / "data" / "style_mix_v1"
REPEAT = 300


def norm(message):
    return " ".join(message.casefold().split())


def read(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def main():
    build_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    train = read(BASE / "train.jsonl")
    valid = read(BASE / "valid.jsonl")
    test = read(BASE / "test.jsonl")
    seed = read(SEED / "train.jsonl")
    heldout = {norm(row["user"]) for row in valid + test}
    overlap = [row["user"] for row in seed if norm(row["user"]) in heldout]
    if overlap:
        raise ValueError(f"Dialogue seed overlaps held-out ChatMix prompts: {overlap}")
    with (OUT / "train.jsonl").open("w", encoding="utf-8") as target:
        for row in train:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
        for _ in range(REPEAT):
            for row in seed:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("valid", "test"):
        shutil.copy2(BASE / f"{split}.jsonl", OUT / f"{split}.jsonl")
    manifest = {"base_dataset": "Adivyanta-ChatMix-v2", "seed_dataset": "Adivyanta-Dialogue-Seed-v1",
                "base_train_rows": len(train), "seed_unique_rows": len(seed), "seed_repeat_factor": REPEAT,
                "effective_train_rows": len(train) + REPEAT * len(seed),
                "valid_rows": len(valid), "test_rows": len(test), "heldout_prompt_overlap": 0,
                "train_sha256": hashlib.sha256((OUT / "train.jsonl").read_bytes()).hexdigest(),
                "purpose": "local experimental weighted sampling; repetitions are not unique dataset rows"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
