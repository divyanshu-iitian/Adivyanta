# Focused chat continuation: stopped after validation regression

The user transcript exposed two separate problems: simple app routing failures and weak raw neural chat. The routing failures were fixed and covered by `tests/test_chat_regression.py`.

For the neural model, `prepare_chat_focus.py` selected 15,652 training and 589 validation examples from ChatMix v2 using source quotas, length limits, and context-dependence/boilerplate filters. This was a **trial dataset**, not a new public release. The same 46M scratch model continued from the ChatMix v2 checkpoint at step 17,000 on an RTX 2050. The trial used batch size 8, gradient accumulation 2, learning rate 0.00006, and a continuation-relative warmup/cosine schedule. The trial command was:

```powershell
.\.venv\Scripts\python.exe prepare_chat_focus.py
.\.venv\Scripts\python.exe train_indic.py --steps 19000 --init-checkpoint adivyanta_indic\chatmix_v2_refined\best.pt --data-dir data\chat_focus --out-dir adivyanta_indic\chat_focus --batch-size 8 --accum 2 --eval-every 250 --lr 0.00006 --relative-schedule
```

| Optimizer step | Focus validation response-token loss |
| --- | ---: |
| 17,000, before trial | **2.3188** |
| 17,250 | 2.3548 |
| 17,500 | 2.3726 |
| 17,750 | 2.3618 |
| 18,000 | 2.3391 |

All four evaluations were worse than the starting checkpoint, so training was stopped at step 18,000. `adivyanta_indic/chat_focus/training_metrics.json` is the full record. The best checkpoint remained the original ChatMix v2 checkpoint. A spot audit also found translated task templates and questions missing context in the selected data. This trial provides no evidence of improved reasoning or chat quality; its weights and dataset were not published as an upgrade.

The app now uses the previously published ChatMix v2 checkpoint by default when present because its earlier held-out English, Hindi, and Hinglish losses were lower than those of the previous chat model. This default choice does **not** imply GPT-1-level conversational ability. App replies based on saved facts, identity, and exact arithmetic are implemented outside the model and are reported separately from neural results.
