"""Publish the reproducible Adivyanta ChatMix v2 dataset after local curation."""
import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi, get_token

from prepare_chatmix_v2 import OUT, ROOT


PACKAGE = ROOT / "data" / "chatmix_v2_upload"


def card(repo_id, manifest):
    counts = manifest["splits"]
    source_lines = "\n".join(
        f"| `{source}` | {amount:,} | {manifest['license_by_source'][source]} |"
        for source, amount in sorted(manifest["train_by_source"].items())
    )
    return f"""---
pretty_name: Adivyanta ChatMix v2
language:
- en
- hi
- bn
- gu
- kn
- ml
- mr
- pa
- ta
- te
- ur
license: other
license_name: mixed-apache-2.0-mit
license_link: https://huggingface.co/datasets/{repo_id}/blob/main/LICENSE_SOURCES.md
task_categories:
- text-generation
tags:
- chat
- hinglish
- multilingual
- curated
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: valid.jsonl
  - split: test
    path: test.jsonl
---

# Adivyanta ChatMix v2

This is a **curated prompt/response dataset**, not a claim that every answer is correct. It has {counts['train']:,} train, {counts['valid']:,} validation and {counts['test']:,} test rows. The data was selected from pinned, licensed source datasets and project-generated arithmetic examples to continue training the scratch Adivyanta GPT. See [the preparation script](https://github.com/divyanshu-iitian/Adivyanta/blob/main/prepare_chatmix_v2.py) and `manifest.json` for reproducibility.

## Schema

Each JSONL row contains `user`, `assistant`, `language`, `source`, `source_license`, and `synthetic`. `synthetic` identifies sources known to be synthetic or project generated; it is **not** a universal guarantee that all other sources are human authored. Do not treat all rows as verified facts.

## Selection

- Source quotas reduce domination by translated/templated Aya shards.
- Exact normalized prompt deduplication spans all included sources and splits.
- The prompt hash assigns new OASST2 and generated arithmetic examples to stable splits; existing held-out splits remain held out.
- Filters remove obvious templates, boilerplate answers, private-looking strings, replacement characters, and examples too long for Adivyanta's 256-token context.
- OASST2 assistant rows are selected from reviewed, non-deleted, non-synthetic replies, with basic ranking and label filters.

These are heuristic filters. They can miss bad answers, bias, private details, and near duplicates. Human review and stronger decontamination are required before high-stakes use. The previous expanded corpus had 254,828 training pairs; this set is intentionally smaller and more focused. It is **not** a comprehensive web or Reddit dataset.

## Training source composition

| Source | Train rows | Source license |
| --- | ---: | --- |
{source_lines}

The source names map to repositories listed in `LICENSE_SOURCES.md`; exact source revisions for earlier data are in [Adivyanta's first](https://github.com/divyanshu-iitian/Adivyanta/blob/main/adivyanta_indic/data_manifest.json) and [expanded](https://github.com/divyanshu-iitian/Adivyanta/blob/main/adivyanta_indic/expanded_data_manifest.json) manifests. OASST2 is pinned in this dataset's manifest. We excluded sources whose dataset cards did not provide clear reusable licensing, including several tempting Hinglish conversation sets.

## Limitations

Source cards supply dataset-level licenses; a redistributed row still inherits its original source terms. License and attribution details are in `LICENSE_SOURCES.md`, and per-row `source_license` is provided for convenience. Generated and translated examples can contain factual errors. Filtering for model context length excludes many valuable long discussions. This dataset cannot by itself make a 46M model comparable to a large general assistant.
"""


def license_text():
    return """# Source licenses and attribution

This is a mixed-source dataset. The selection script and generated arithmetic examples are original MIT-licensed project work. Other rows retain their source terms:

| `source` values | Source dataset | Source card license |
| --- | --- | --- |
| `aya_human` | https://huggingface.co/datasets/CohereLabs/aya_collection | Apache-2.0 |
| `aya_hindi`, `aya_hindi_extra`, `aya_english` | https://huggingface.co/datasets/CohereLabs/aya_collection_language_split | Apache-2.0 |
| `oasst` | https://huggingface.co/datasets/OpenAssistant/oasst1 | Apache-2.0 |
| `oasst2` | https://huggingface.co/datasets/OpenAssistant/oasst2 | Apache-2.0 |
| `hinglish` | https://huggingface.co/datasets/DSMJ910/hinglish-instruct-10k | Apache-2.0 |
| `ultrachat` | https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k | MIT |
| `gsm8k` | https://huggingface.co/datasets/openai/gsm8k | MIT |
| `original`, `generated_math` | Adivyanta project | MIT |

Check the upstream cards and pinned manifests before redistributing or using these data commercially. The original datasets and their authors should be credited in derived work.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", help="Default: <logged-in-user>/Adivyanta-ChatMix-v2")
    args = parser.parse_args()
    if not get_token():
        raise SystemExit("Hugging Face login required")
    api = HfApi()
    repo_id = args.repo_id or f"{api.whoami()['name']}/Adivyanta-ChatMix-v2"
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for filename in ("train.jsonl", "valid.jsonl", "test.jsonl", "manifest.json"):
        shutil.copy2(OUT / filename, PACKAGE / filename)
    (PACKAGE / "README.md").write_text(card(repo_id, manifest), encoding="utf-8")
    (PACKAGE / "LICENSE_SOURCES.md").write_text(license_text(), encoding="utf-8")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(PACKAGE),
                      commit_message="Publish curated Adivyanta ChatMix v2 with provenance and splits")
    print(f"https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
