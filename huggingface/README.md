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
---

# Adivyanta 46M

Adivyanta is a **46,349,312-parameter decoder-only GPT trained from random initialization** for experimental English, Hindi and Roman Hindi chat. It uses its own 16,384-token byte-level BPE tokenizer. No pretrained model or tokenizer weights were used.

**This is a research/learning release, not a dependable assistant.** Its raw answers often drift off topic, invent facts or produce malformed Hindi. It has not demonstrated sentience. It has no measured advantage over Qwen or other established assistants. The project's app adds local memory, a calculator and a few authored replies, none of which are part of these model weights.

## Files

- `model.safetensors`: best expanded **base** checkpoint, selected on mixed-source validation. Prefer this for English or Devanagari experiments.
- `chat_model.safetensors`: continuation of the base for 3,000 conversational steps. Hinglish test loss is lower, while English and Hindi test loss are higher.
- `config.json`, `tokenizer.json`, `model.py`, `inference.py`: custom PyTorch inference. This is **not a Transformers AutoModel checkpoint** and the Hugging Face inference widget may not run it.
- `metrics/`: validation and held-out test metrics; `benchmarks/`: fixed raw model-only outputs.

## Try it

Install PyTorch, `tokenizers==0.22.2`, `safetensors`, and `huggingface_hub`. Then download this model repository and run:

```powershell
python inference.py --prompt "Kya scene hai?" --variant chat
python inference.py --prompt "Explain what a compiler does." --variant base
```

`inference.py` loads only local files. It samples a response and does not add the app's memory or authored replies. You can compare both variants with the same prompt and seed.

## Training and data

The architecture has 12 decoder layers, 8 attention heads, 512 hidden dimensions, tied input/output embeddings, and a 256-token context. The first 6,000 steps used 120,449 selected training pairs. Training then continued to 12,000 total steps on an expanded 254,828-pair training split. The chat variant received 3,000 additional steps on conversational examples. All weights in both files descend from the same random initialization.

Sources are Aya Collection, OpenAssistant OASST1, Hinglish Instruct 10K, UltraChat 200K and GSM8K, plus 16 original examples. The selected source cards state Apache-2.0 or MIT licenses. Dataset versions, counts, filtering and split code are in the [GitHub source repository](https://github.com/divyanshu-iitian/Adivyanta), especially `DATASETS.md`, `prepare_indic_data.py`, and `prepare_expanded_data.py`. Raw datasets are not redistributed here. The official GSM8K test questions were excluded from training.

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

## License and limitations

The original code and these scratch Indic weights are MIT licensed; see `LICENSE`. The older 9.71M checkpoints from the GitHub project used Meta EmpatheticDialogues under CC BY-NC 4.0 and are **not included here**.

Training data includes translated, templated and synthetic answers and may contain errors or social bias. The model can hallucinate, repeat, mix languages, and mishandle sensitive queries. Use source verification and human review for factual or consequential tasks.
