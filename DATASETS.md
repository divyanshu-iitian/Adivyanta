# Adivyanta data ledger

This project keeps the prepared corpus locally under `data/indic/` and `data/indic_expanded/`. Both directories are ignored by Git. The public repository contains the preparation scripts, pinned source revisions, manifests, and model checkpoints. Run `prepare_indic_data.py` and then `prepare_expanded_data.py` to rebuild the same selection.

| Source | Source card | License shown on card | Selected use |
| --- | --- | --- | --- |
| Aya Collection human examples | [CohereLabs/aya_collection](https://huggingface.co/datasets/CohereLabs/aya_collection) | Apache-2.0 | Indian-language and English instructions |
| Aya Hindi and English shards | [CohereLabs/aya_collection_language_split](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | Apache-2.0 | Hindi and English coverage |
| OASST1 | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Apache-2.0 | Human conversation turns |
| Hinglish Instruct 10K | [DSMJ910/hinglish-instruct-10k](https://huggingface.co/datasets/DSMJ910/hinglish-instruct-10k) | Apache-2.0 | Roman Hindi conversation |
| UltraChat 200K | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | Longer English assistant turns |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | MIT | Worked arithmetic word problems; official test kept out of training |

The first build has 120,449 train, 1,155 validation, and 1,200 test pairs. The expanded build has 254,828 train, 2,460 validation, and 2,515 test pairs after prompt deduplication across the included sources. These counts are model examples, not unique human conversations. The exact source revisions and counts are in [the initial manifest](adivyanta_indic/data_manifest.json) and [the expanded manifest](adivyanta_indic/expanded_data_manifest.json).

The source mix is a deliberate subset. More rows are not automatically better: duplicate prompts, translated templates, synthetic replies, and poor answers can hurt the model. These selections still contain such flaws. There is no claim that every available public dataset is licensed, relevant, or included. Reddit posts were not collected. Before adding another source, check its terms, provenance, quality, and whether it duplicates a held-out question. Keep the raw source license and attribution with any redistributed corpus.

Data splits use a deterministic hash of the normalized prompt so repeated prompts stay in one split. Cross-source duplicate prompts are removed before the expanded split is written. The 16,384-token byte-level BPE tokenizer was trained only on the original training split. The expanded model reuses it. The math benchmark separately downloads the official GSM8K test split and never adds it to training.
