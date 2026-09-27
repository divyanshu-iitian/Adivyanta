"""Publish the project-authored dialogue seed without any local user memory."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi, get_token

from prepare_dialogue_seed_v1 import OUT, ROOT, main as build


PACKAGE = ROOT / "data" / "dialogue_seed_v1_upload"


def card(repo_id, count):
    return f"""---
pretty_name: Adivyanta Dialogue Seed v1
language:
- en
- hi
license: mit
task_categories:
- text-generation
tags:
- dialogue
- hinglish
- roast
- synthetic
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
---

# Adivyanta Dialogue Seed v1

This is a **small, project-authored synthetic style supplement** with {count} English, Hindi, and Hinglish prompt/response examples. It covers casual chat, local-memory wording, honest uncertainty, dating advice, and consensual, sometimes profane roasts. It is **not** a general knowledge corpus, a benchmark, or proof of model improvement. There is no held-out split because the set is too small to support a meaningful independent estimate.

The data contains no exported user chats or local memory. Names in examples are fictional. The examples were authored for this project, not scraped from social media. Some rows contain Hindi slang and profanity; they target habits or the bot in opt-in banter, not protected groups. The rows are MIT-licensed project work.

Each `train.jsonl` row has `user`, `assistant`, `language`, `intent`, `source`, `source_license`, and `synthetic`. `manifest.json` includes counts and a SHA-256 checksum. Rebuild with [prepare_dialogue_seed_v1.py](https://github.com/divyanshu-iitian/Adivyanta/blob/main/prepare_dialogue_seed_v1.py). Review examples before mixing into a training run. The larger [ChatMix v2 dataset](https://huggingface.co/datasets/{repo_id.rsplit('/', 1)[0]}/Adivyanta-ChatMix-v2) is separate and retains its upstream source licenses.

The Adivyanta app uses deterministic memory and authored replies for some matching prompts. Publishing these {count} examples does **not** mean the current 46M model weights were retrained on them.
"""


LICENSE = """MIT License

Copyright (c) 2026 Adivyanta contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this dataset and associated documentation files (the "Dataset"), to deal
in the Dataset without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Dataset, and to permit persons to whom the Dataset is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Dataset.

THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE
DATASET.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", help="Default: <logged-in-user>/Adivyanta-Dialogue-Seed-v1")
    args = parser.parse_args()
    if not get_token():
        raise SystemExit("Hugging Face login required")
    build()
    api = HfApi()
    repo_id = args.repo_id or f"{api.whoami()['name']}/Adivyanta-Dialogue-Seed-v1"
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for filename in ("train.jsonl", "manifest.json"):
        shutil.copy2(OUT / filename, PACKAGE / filename)
    (PACKAGE / "README.md").write_text(card(repo_id, manifest["rows"]), encoding="utf-8")
    (PACKAGE / "LICENSE").write_text(LICENSE, encoding="utf-8")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(PACKAGE),
                      commit_message="Publish versioned Adivyanta dialogue style seed with provenance")
    print(f"https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
