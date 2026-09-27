---
language:
- en
- hi
license: mit
pipeline_tag: text-generation
tags:
- from-scratch
- gpt
- hinglish
- experimental
datasets:
- CohereLabs/aya_collection
- CohereLabs/aya_collection_language_split
- OpenAssistant/oasst1
- DSMJ910/hinglish-instruct-10k
- HuggingFaceH4/ultrachat_200k
- openai/gsm8k
- OpenAssistant/oasst2
---

# Adivyanta 46M

Adivyanta is a **46,349,312-parameter decoder-only GPT trained from random initialization** for experimental English, Hindi and Roman Hindi chat. It uses its own 16,384-token byte-level BPE tokenizer. No pretrained model or tokenizer weights were used.

**This is a research/learning release, not a dependable assistant.** Its raw answers often drift off topic, invent facts or produce malformed Hindi. It has not demonstrated sentience. It has no measured advantage over Qwen or other established assistants. The project's app adds local memory, a calculator and a few authored replies, none of which are part of these model weights.

The newer **ChatMix v2** checkpoint was trained for 5,000 more optimizer steps on the [public curated dataset](https://huggingface.co/datasets/divyanshumishra/Adivyanta-ChatMix-v2). It fits held-out text better than the earlier chat checkpoint, but its actual answers remain unreliable. It scored **0/100** on generated held-out arithmetic and **0/50** on a fixed official GSM8K test subset. It is released for transparent experiments, not as a quality upgrade for everyday use.

## Files

- `model.safetensors`: best expanded **base** checkpoint, selected on mixed-source validation. Prefer this for English or Devanagari experiments.
- `chat_model.safetensors`: continuation of the base for 3,000 conversational steps. Hinglish test loss is lower, while English and Hindi test loss are higher.
- `chatmix_v2_model.safetensors`: continuation of the chat variant for 5,000 further steps, ending at optimizer step 17,000. Selected by ChatMix v2 validation loss.
- `config.json`, `tokenizer.json`, `model.py`, `inference.py`: custom PyTorch inference. This is **not a Transformers AutoModel checkpoint** and the Hugging Face inference widget may not run it.
- `metrics/`: validation and held-out test metrics; `benchmarks/`: fixed raw model-only outputs.

## Try it

Install PyTorch, `tokenizers==0.22.2`, `safetensors`, and `huggingface_hub`. Then download this model repository and run:

```powershell
python inference.py --prompt "Kya scene hai?" --variant chat
python inference.py --prompt "Explain what a compiler does." --variant base
python inference.py --prompt "Kya scene hai?" --variant chatmix-v2
```

`inference.py` loads only local files. It samples a response and does not add the app's memory or authored replies. You can compare variants with the same prompt and seed.

## Training and data

The architecture has 12 decoder layers, 8 attention heads, 512 hidden dimensions, tied input/output embeddings, and a 256-token context. The first 6,000 steps used 120,449 selected training pairs. Training then continued to 12,000 total steps on an expanded 254,828-pair training split. The chat variant received 3,000 additional steps on conversational examples. All weights in both files descend from the same random initialization.

Sources are Aya Collection, OpenAssistant OASST1, Hinglish Instruct 10K, UltraChat 200K and GSM8K, plus 16 original examples. The selected source cards state Apache-2.0 or MIT licenses. Dataset versions, counts, filtering and split code are in the [GitHub source repository](https://github.com/divyanshu-iitian/Adivyanta), especially `DATASETS.md`, `prepare_indic_data.py`, and `prepare_expanded_data.py`. Raw datasets are not redistributed here. The official GSM8K test questions were excluded from training.

The ChatMix v2 continuation used [92,360 filtered training pairs](https://huggingface.co/datasets/divyanshumishra/Adivyanta-ChatMix-v2), including OASST2 and 10,000 generated arithmetic pairs. Validation and test splits contain 974 and 964 pairs. Each published row has source and license fields; the dataset card describes filtering and remaining quality risks. This checkpoint has the same architecture and tokenizer and descends from the original random initialization. No external pretrained weights were loaded.

## Evaluation

| Measurement | Base | Chat variant |
| --- | ---: | ---: |
| Expanded validation response-token loss | 3.0588 | — |
| 100-example chat validation response-token loss | 4.5346 | 3.9881 |
| English test loss, 884 examples | 4.3479 | 4.6052 |
| Hindi test loss, 1,391 examples | 2.0542 | 2.5803 |
| Hinglish test loss, 42 examples | 3.7901 | 3.0491 |
| Official GSM8K test, seeded 50-question exact-number subset | — | **0/50** |

Loss is next-token fit to the selected source distribution, not a measure of truth or usefulness. The Hinglish estimate is based on only 42 examples. The [full benchmark report](https://github.com/divyanshu-iitian/Adivyanta/blob/main/benchmarks/REPORT.md) and raw model-only outputs show severe reasoning and factual limitations. In particular, the GSM8K result does **not** support a claim of reliable math or logic.

### ChatMix v2 continuation

| Measurement | Previous chat | ChatMix v2 |
| --- | ---: | ---: |
| Earlier expanded test English response-token loss (884 examples) | 4.6052 | 4.2042 |
| Earlier expanded test Hindi response-token loss (1,391 examples) | 2.5803 | 2.0209 |
| Earlier expanded test Hinglish response-token loss (42 examples) | 3.0491 | 2.9945 |
| ChatMix v2 validation loss | 3.4646 at start of continuation | 2.8445 at step 17,000 |
| Held-out generated arithmetic exact answers (100 questions) | 0/100 | 0/100 |
| Official GSM8K seeded test subset exact answers (50 questions) | 0/50 | 0/50 |

The legacy test losses use the **same test set** for both checkpoints. ChatMix v2's own 964-row test losses and raw sampled/greedy answers are included in `metrics/` and `benchmarks/`. The ChatMix validation row compares the v2 checkpoint before and after its continuation, with the same validation split. Lower loss did **not** make it a dependable reasoner, conversationalist, or roast writer. See the [ChatMix v2 benchmark report](https://github.com/divyanshu-iitian/Adivyanta/blob/main/benchmarks/CHATMIX_V2_REPORT.md).

A later [focused chat continuation trial](https://github.com/divyanshu-iitian/Adivyanta/blob/main/benchmarks/CHAT_FOCUS_TRIAL.md) made validation loss worse at every measured checkpoint through step 18,000, so its weights were not published as an upgrade. The current ChatMix v2 file remains the best selected checkpoint from these experiments, despite its poor raw replies.

## License and limitations

The original code and these scratch Indic weights are MIT licensed; see `LICENSE`. The older 9.71M checkpoints from the GitHub project used Meta EmpatheticDialogues under CC BY-NC 4.0 and are **not included here**.

Training data includes translated, templated and synthetic answers and may contain errors or social bias. The model can hallucinate, repeat, mix languages, and mishandle sensitive queries. Use source verification and human review for factual or consequential tasks.
