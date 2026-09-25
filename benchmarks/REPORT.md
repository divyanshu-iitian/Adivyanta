# Adivyanta 46M evaluation

These results use the scratch-trained 46,349,312-parameter model. All outputs below are from the neural model alone: app memory, calculator, authored chat openers, and authored roast lines are disabled.

| Checkpoint | Measured split | Metric | Result |
| --- | --- | --- | ---: |
| Initial scratch base, before expanded continuation | Expanded validation | Response-token loss | 3.4672 |
| Expanded base, 12,000 total optimizer steps | Expanded validation | Response-token loss | 3.0588 |
| Expanded base, before chat tuning | 100 held-out conversation examples | Response-token loss | 4.5346 |
| Expanded chat, 3,000 additional steps | Same 100 conversation examples | Response-token loss | 3.9881 |
| Expanded chat | Official GSM8K test, fixed seeded 50-question subset | Exact numeric answers | **0/50 (0%)** |

The expanded 2,515-example test split is source mixed. Below are response-token losses on languages with at least 40 examples; lower is better. The base and chat checkpoints use the **same** test rows.

| Language | Test examples | Expanded base | Expanded chat |
| --- | ---: | ---: | ---: |
| English | 884 | 4.3479 | 4.6052 |
| Hindi | 1,391 | 2.0542 | 2.5803 |
| Hinglish | 42 | 3.7901 | 3.0491 |
| Tamil | 64 | 2.5646 | 2.4729 |
| Telugu | 51 | 3.2215 | 3.1457 |

The Hinglish estimate has only 42 examples. Other Indian-language groups have even fewer test examples. These losses show fit to the selected dataset distributions, not correctness. Chat tuning helped the focused conversational set and Hinglish, while harming English and Hindi test loss. The full per-language results are in [`base_test_metrics.json`](../adivyanta_indic/expanded/base_test_metrics.json) and [`test_metrics.json`](../adivyanta_indic/expanded/test_metrics.json).

Read the fixed 16-prompt outputs from the [first 6,000-step model](indic_model_outputs_6000.jsonl), [expanded base](indic_expanded_base_outputs.jsonl), and [expanded chat checkpoint](indic_expanded_outputs.jsonl). The expanded chat model often gives off-topic English and Hinglish replies, malformed Hindi, and weak model-only roasts. The math result is in [`math_summary.json`](math_summary.json), with each question and answer in [`math_outputs.jsonl`](math_outputs.jsonl). The sample is small and the last generated number is counted as the answer.

**Conclusion:** the continued training improved measured validation fit, but the released 46M model is not a reliable logical or factual assistant. There is no evidence it outperforms Qwen or another established model. The chat app handles a few common openers, local user facts, explicit corrections, simple arithmetic, and authored roast requests separately from model generation; those features must not be confused with model-only ability.
