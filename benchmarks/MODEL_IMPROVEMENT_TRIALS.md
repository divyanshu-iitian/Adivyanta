# Local model improvement trials, 28 September 2026

These are **raw 46M GPT checkpoint** trials on the local RTX 2050 (4 GB). They do not include app memory, authored replies, the calculator, or roast routing. Neither trial made Adivyanta a dependable conversational model, so the released default checkpoint remains ChatMix v2 step 17,000.

## Fixed behavioral probe

`benchmark_dialogue_probes.py` runs 12 novel English and Hinglish prompts through the neural checkpoint, with fixed sampling seeds, temperature 0.75, top-k 30, and a 64-token output limit. It uses fictional names and no private user transcript. This is a small diagnostic, not a calibrated capability score. Compare the complete outputs:

- [Starting ChatMix v2 checkpoint](dialogue_probe_baseline.jsonl)
- [Weighted style mix](dialogue_probe_style_v1.jsonl), plus its [greedy decode](dialogue_probe_style_v1_greedy.jsonl)
- [Seed-only specialization](dialogue_probe_persona_trial.jsonl)

The starting model answered “My girlfriend is Neha.” with an unrelated invented description of language models. The style mix replied “Palic.” on the same prompt. Across the probe, both continuation checkpoints still produced unrelated, broken, or empty-looking replies. One sampled style-mix reply preferred “Samosa!” for the burger-or-samosa prompt; that isolated success does not establish general improvement. Greedy decoding did not make the style checkpoint consistently coherent.

## Training and held-out loss

| Trial | Data | Steps from base | ChatMix validation response-token loss | Outcome |
| --- | --- | ---: | ---: | --- |
| Starting checkpoint | ChatMix v2 | 17,000 | 2.8445 | Released experimental baseline |
| Weighted style mix | 92,360 ChatMix rows and 66 Dialogue Seed rows each repeated 300 times | 18,500 | 2.8253 | Slight loss improvement, raw chat still poor |
| Seed-only specialization | 66 Dialogue Seed rows | 17,500 | 3.0375 | Overfit and regressed |

The weighted mix has 112,160 **effective training rows**, but only 66 unique new rows. Repetition is not new data. Its local manifest records the source counts and SHA-256. ChatMix validation and test splits were copied unchanged; no normalized seed prompt overlaps those splits. The seed-only trial reached training loss 0.4144 while held-out validation worsened, consistent with memorization.

The weighted checkpoint's [test metrics](../adivyanta_indic/style_mix_v1/test_metrics.json) use the same 964 ChatMix test rows as the [baseline](../adivyanta_indic/chatmix_v2_refined/test_metrics.json):

| Language | Examples | Starting loss | Weighted loss |
| --- | ---: | ---: | ---: |
| English | 345 | 4.1114 | 4.1066 |
| Hindi | 463 | 1.7575 | 1.7323 |
| Hinglish | 54 | 2.9648 | 2.9921 |

The small English and Hindi changes measure reference-token prediction, not usefulness. Hinglish worsened. Other language samples are too small for firm conclusions. [Training metrics](../adivyanta_indic/style_mix_v1/training_metrics.json) and [seed-only metrics](../adivyanta_indic/persona_trial/training_metrics.json) record the full steps. The seed-only `best.pt` stayed at the original step 17,000; the final overfit weights were exported solely for the raw probe and are not released.

## Reproduce locally

Run after preparing ChatMix v2 and its checkpoint:

```powershell
.\.venv\Scripts\python.exe benchmark_dialogue_probes.py --checkpoint adivyanta_indic\chatmix_v2_refined\best.pt --out benchmarks\dialogue_probe_baseline.jsonl
.\.venv\Scripts\python.exe prepare_style_mix_v1.py
.\.venv\Scripts\python.exe train_indic.py --steps 18500 --init-checkpoint adivyanta_indic\chatmix_v2_refined\best.pt --data-dir data\style_mix_v1 --out-dir adivyanta_indic\style_mix_v1 --batch-size 8 --accum 2 --eval-every 250 --lr 0.00002 --relative-schedule
.\.venv\Scripts\python.exe benchmark_dialogue_probes.py --checkpoint adivyanta_indic\style_mix_v1\best.pt --out benchmarks\dialogue_probe_style_v1.jsonl
.\.venv\Scripts\python.exe prepare_persona_trial.py
.\.venv\Scripts\python.exe train_indic.py --steps 17500 --init-checkpoint adivyanta_indic\chatmix_v2_refined\best.pt --data-dir data\persona_trial --out-dir adivyanta_indic\persona_trial --batch-size 8 --accum 2 --eval-every 250 --lr 0.00001 --relative-schedule
.\.venv\Scripts\python.exe snapshot_training_step.py --last adivyanta_indic\persona_trial\last.pt --template adivyanta_indic\chatmix_v2_refined\best.pt --out adivyanta_indic\persona_trial\trial_step_17500.pt
.\.venv\Scripts\python.exe benchmark_dialogue_probes.py --checkpoint adivyanta_indic\persona_trial\trial_step_17500.pt --out benchmarks\dialogue_probe_persona_trial.jsonl
```

The weighted trial took 317 seconds for 1,500 steps; the seed-only trial took 68 seconds for 500 steps. These timings include periodic validation and checkpoint writes on one local machine, so they are not a general throughput benchmark. Trial checkpoints and local data are excluded from Git. No trial weights were promoted to the app or Hugging Face model repo.

## What the result says

The model's current training objective predicts the assistant part of individual prompt/reply records. It has no broad raw-text language-modeling phase and no neural multi-turn conversation training. The limited prompt context and mixed-quality source answers also constrain results. More repetitions of 66 authored rows cannot supply broad knowledge or reasoning. A credible improvement needs a much larger, carefully licensed and filtered English/Hindi/Hinglish text corpus, sustained pretraining on the local GPU, then conversation tuning and fresh held-out probes. A local 4 GB GPU can run experiments, but there is no basis to promise ChatGPT- or Qwen-level quality from these runs.
