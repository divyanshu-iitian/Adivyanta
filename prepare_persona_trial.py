"""Local seed-only specialization trial; keeps ChatMix validation/test untouched."""
import json
from pathlib import Path
import shutil

from prepare_dialogue_seed_v1 import main as build_seed


ROOT = Path(__file__).parent
OUT = ROOT / "data" / "persona_trial"


def main():
    build_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "data" / "dialogue_seed_v1" / "train.jsonl", OUT / "train.jsonl")
    for split in ("valid", "test"):
        shutil.copy2(ROOT / "data" / "chatmix_v2" / f"{split}.jsonl", OUT / f"{split}.jsonl")
    manifest = {"train": "Adivyanta-Dialogue-Seed-v1", "train_unique_rows": 66,
                "validation": "Adivyanta-ChatMix-v2 validation", "test": "Adivyanta-ChatMix-v2 test",
                "purpose": "experimental narrow specialization; not a dataset release"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
