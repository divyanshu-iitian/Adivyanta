# Adivyanta ChatMix v2: measured results

The 46,349,312-parameter GPT was initialized from random weights, trained on the first and expanded datasets, then conversationally tuned. The new checkpoint continues that **same scratch lineage** from optimizer step 12,000 to 17,000 on [ChatMix v2](https://huggingface.co/datasets/divyanshumishra/Adivyanta-ChatMix-v2). It is an experimental release and is still not a dependable chat assistant.

## Dataset and procedure

The published filtered splits contain 92,360 train, 974 validation, and 964 test prompt/reply rows. `prepare_chatmix_v2.py` records source revisions, quotas, filtering, split handling, and generated arithmetic. `validate_chatmix_v2.py` checks duplicate prompts across splits and generated arithmetic answers. The Hugging Face dataset card and each row record provenance and license information. The data may still contain weak, incorrect, translated, or templated answers.

Training used the existing 16,384-token BPE tokenizer and 256-token context, batch size 8 with gradient accumulation 2, learning rate 0.00015, and validation every 500 steps on an RTX 2050 4 GB GPU. The best checkpoint was the final step, with ChatMix validation response-token loss falling from 3.4646 to 2.8445. See `adivyanta_indic/chatmix_v2_refined/training_metrics.json` for the recorded evaluations.

## Held-out measures

| Measure | Previous chat | ChatMix v2 |
| --- | ---: | ---: |
| Earlier expanded test English loss, 884 examples | 4.6052 | 4.2042 |
| Earlier expanded test Hindi loss, 1,391 examples | 2.5803 | 2.0209 |
| Earlier expanded test Hinglish loss, 42 examples | 3.0491 | 2.9945 |
| Generated arithmetic test, exact answer, 100 questions | 0/100 | 0/100 |
| Official GSM8K test, seeded 50-question subset, exact answer | 0/50 | 0/50 |

The ChatMix v2 checkpoint's own 964-example test losses are in `adivyanta_indic/chatmix_v2_refined/test_metrics.json`. The comparison losses above use the **same older expanded test split** for both models. These are response-token prediction losses, so they measure fit to reference text rather than response correctness or human preference. The Hinglish sample is especially small. The arithmetic benchmark and official GSM8K subset are narrow; GSM8K scoring extracts the final generated number. The full model-only outputs and benchmark summaries are in this directory.

On the 16 fixed conversation prompts, sampled and greedy outputs still include unrelated replies, gibberish, broken Hindi, and failed roasts. Greedy decoding did not resolve these problems. Read `indic_chatmix_v2_outputs.jsonl` and `indic_chatmix_v2_greedy_outputs.jsonl` before relying on the model. The local app's memory, canned replies, and calculator are separate code and are **not** included in the neural benchmark scores.

There is no head-to-head quality evaluation establishing superiority over Qwen, ChatGPT, or another established assistant. The poor raw replies and 0/50 GSM8K result give no basis for that claim. This release makes the dataset, training lineage, checkpoint, and limitations inspectable; a substantially larger, better reviewed corpus and much more compute would be needed for a competitive general assistant.
