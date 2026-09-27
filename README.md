# Adivyanta: an India-focused GPT trained from scratch

The model is also packaged for Hugging Face with `export_huggingface.py` and `publish_huggingface.py`. The export contains standalone `safetensors` weights, tokenizer, PyTorch inference code, provenance, and the [model card](huggingface/README.md); the prepared upload folder stays under ignored `data/huggingface_upload/`.

[Hinglish training walkthrough](TRAINING_GUIDE_HI.md) explains how the dataset, tokenizer, GPT weights, validation, and chat memory were built.

Adivyanta's default chat model is a **46,349,312-parameter decoder-only Transformer initialized at random**. Its tokenizer was trained from the project data. It loads **no pretrained model weights**. The goal is local English, Hindi, and Roman Hindi (Hinglish) chat and playful roasts. The smaller 9.71M scratch GPT remains available as a historical baseline; the previous borrowed-base adapter has been removed from the current project.

This is a small experimental model, not a ChatGPT-level assistant or a general model for every Indian language. Its responses can repeat, drift, or be wrong. Persistent memory improves continuity, but memory is separate from the model weights. A next-token predictor has no demonstrated sentience.

## Run locally

Python 3.11+ and PyTorch are required. For NVIDIA CUDA, choose the matching [official PyTorch package](https://pytorch.org/get-started/locally/); CPU inference also works but is slower. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe chat.py
```

Type `/quit` to leave. The model and tokenizer are in `adivyanta_indic/`; starting chat does not download any model. A one-shot example:

```powershell
.\.venv\Scripts\python.exe chat.py --prompt "kya kar rahe ho"
```

The first 9.71M GPT can still be tried with `chat.py --scratch`. Its English-only corpus and 128-token context make it considerably less capable in Hindi.

### Memory and learning from corrections

The interactive app saves explicit user facts, the last 40 turns, and corrections in `data/adivyanta_memory.json` on **your own computer**. The `data/` directory is excluded from Git. It remembers statements such as `my name is Rohan`, `mera naam Rohan hai`, `Riya is my gf`, `I prefer Hinglish`, and `remember that I like samosas`. Names and exact corrections are retrieved directly, so they persist after closing and reopening chat.

The app also evaluates simple arithmetic expressions exactly, for example `what is 23 * (4 + 2)?` → `138`. This answer comes from a small calculator, not from the GPT weights. Common openers such as `hi`, `samosa`, and `kya kar rahe ho` have authored replies. These grounded features reduce errors on supported requests; they do not solve general hallucination.

The app recognizes more direct Hindi/Hinglish phrasing, such as `Riya meri gf ka naam hai`, `tum Riya ko jante ho?`, and `tera naam kya hai?`. It uses the local fact you supplied and does not claim to know Riya personally. On explicit requests such as `roast my code` or `gaali de`, it can give playful profane roasts. Casual messages are not treated as consent to roast. These are app responses, not evidence that the neural checkpoint learned the same behavior.

For `trip panchmadi ki hai` followed by `yes make it`, the app keeps a short-lived local task state and gives a starter Pachmarhi itinerary. The listed places are from [Madhya Pradesh Tourism](https://www.mptourism.com/destination-panchmarhi.php); the app does not check live weather, entry rules, availability, or prices. It asks for trip length and origin to refine the outline. This is a narrow offline planner, not general travel expertise learned by the GPT weights.

When the ChatMix v2 checkpoint is present, the app uses it by default because its held-out response-token losses were lower than the previous chat checkpoint across English, Hindi, and Hinglish. If it is absent, the app uses the earlier conversational checkpoint for Roman Hindi and the expanded base checkpoint for English or Devanagari. All are this project's scratch weights. `--checkpoint` forces one chosen checkpoint, and a local personalized checkpoint takes priority if you explicitly trained it. Lower loss has **not** made free-form neural answers consistently coherent; the calculator, identity answers, and explicitly stored facts use separate app logic.

The released checkpoints were trained on single user/assistant pairs. Neural generation now receives the current message only; earlier generated replies are no longer concatenated into an untrained multi-turn prompt. The app still stores up to 40 turns for local inspection, and explicit facts/corrections remain available through memory routing. Proper neural multi-turn context requires a separate training and evaluation run.

Commands:

| Command | Effect |
| --- | --- |
| `/memory` | Inspect saved facts and memory counts |
| `/remember I like samosas` | Save a note |
| `/correct kya kar rahe ho => Tumse baat kar raha hoon.` | Reuse this answer when that question is asked again |
| `/forget name`, `/forget girlfriend`, `/forget trip`, or `/forget all` | Delete selected or all local memory |
| `/help` | Show commands |

Memory and exact-question feedback work immediately. Chatting does not secretly change model weights. After saving corrections, you can **explicitly update your local model weights** with `.\.venv\Scripts\python.exe train_feedback.py`. That short run accepts the updated weights only when correction loss improves and related chat validation loss stays within 0.10 of its starting value. Its `data/personalized.pt` checkpoint stays on your computer; the default chat loads it on its next start. This is limited personalization, not a guarantee of broad intelligence. The memory and personalized weight files are never sent to GitHub.

## Data and training

The new 46M model was initialized randomly and trained only on prompt/response examples from these datasets and 16 original short examples:

| Source | Selected pairs | License stated by source | Role |
| --- | ---: | --- | --- |
| [Aya Collection, human subset](https://huggingface.co/datasets/CohereLabs/aya_collection) | 29,964 | Apache-2.0 | Indian-language and English examples |
| [Aya Collection, Hindi shard](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | 50,000 | Apache-2.0 | Hindi instructions and replies |
| [Aya Collection, English shard](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) | 50,000 | Apache-2.0 | English dialogue-style instructions |
| [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 11,451 | Apache-2.0 | Human chat turns |
| [Hinglish Instruct 10K](https://huggingface.co/datasets/DSMJ910/hinglish-instruct-10k) | 5,780 | Apache-2.0 | Synthetic Roman Hindi and English chat |

Repeated prompts were removed, leaving **120,449 training**, **1,155 validation**, and **1,200 test** examples. The source revisions, counts, and selection rules are in [prepare_indic_data.py](prepare_indic_data.py); the generated manifest is saved under `data/indic/`. The tokenizer has 16,384 byte-level BPE tokens and was trained on **training examples only**. The model has 12 Transformer layers, eight attention heads, 512 hidden dimensions, tied embeddings, and a 256-token context. Training predicts reply tokens using cross-entropy, AdamW, mixed precision, gradient accumulation, clipping, and validation checkpoint selection. Read [train_indic.py](train_indic.py) for the exact procedure.

A second corpus build in [prepare_expanded_data.py](prepare_expanded_data.py) adds **99,937** more [Aya Hindi](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split) examples, **29,818** [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) turns (MIT), and **7,244** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) math examples (MIT), after cross-source prompt deduplication. The expanded split has **254,828 train**, **2,460 validation**, and **2,515 test** examples. [Its pinned manifest](adivyanta_indic/expanded_data_manifest.json) records versions and source counts. The added data is local under `data/indic_expanded/`; the public repo contains the reproducible collection scripts and manifest, not copied raw datasets. The original scratch tokenizer is retained. No pretrained model or tokenizer is loaded.

[DATASETS.md](DATASETS.md) records provenance, licenses, split handling, and the limits of this collection.

A smaller, cleaner [Adivyanta ChatMix v2 dataset](https://huggingface.co/datasets/divyanshumishra/Adivyanta-ChatMix-v2) is now published separately. It is derived from the licensed sources above plus OASST2 and project-generated arithmetic, with 92,360 train / 974 validation / 964 test rows. Run `prepare_chatmix_v2.py` and `validate_chatmix_v2.py` to rebuild and check it. It is a filtered experiment, not a claim that every row is accurate.

[Dialogue Seed v1](https://huggingface.co/datasets/divyanshumishra/Adivyanta-Dialogue-Seed-v1) is a separate 66-row, project-authored style supplement with greetings, memory phrasing, supportive replies, trip-intent wording, conversational repair, dating advice and opt-in roasts. Run `prepare_dialogue_seed_v1.py` to rebuild it and `publish_dialogue_seed_v1.py` to publish it. It contains fictional names and no local user chats. The seed set is too small for a meaningful independent benchmark, and the current model weights were not retrained on it.

An experimental [ChatMix v2 model checkpoint](https://huggingface.co/divyanshumishra/Adivyanta-46M) continues the scratch-trained chat model for 5,000 steps on that dataset. The checkpoint is `chatmix_v2_model.safetensors` and the download includes raw outputs and metrics. Run `python inference.py --variant chatmix-v2 --prompt "Kya scene hai?"` inside the downloaded model folder. The [ChatMix v2 benchmark report](benchmarks/CHATMIX_V2_REPORT.md) compares it with the earlier chat checkpoint: held-out response-token losses improved, but exact arithmetic remained 0/100 and the fixed GSM8K subset remained 0/50. Its free-form replies are still frequently poor.

A later [focused chat continuation trial](benchmarks/CHAT_FOCUS_TRIAL.md) selected more self-contained examples and trained through step 18,000, but all four validation checks were worse than its starting checkpoint. That trial was stopped and was not released as an improved model. The app fixes for arithmetic wording, assistant identity, and explicitly stated relationship facts are separate from neural training.

The corpus includes some other Indian languages through Aya, but their sample counts are small. Hindi, Hinglish, and English are the main targets. The Aya shards include translated or templated examples, and the Hinglish dataset is synthetic; they are not equivalent to a large, carefully edited native conversation corpus. No Reddit posts were scraped. Dataset licenses permit broad reuse according to their cards; inspect the cards and provenance before using the model for a specific product.

To reproduce on a CUDA machine:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe prepare_indic_data.py
.\.venv\Scripts\python.exe train_indic.py --steps 6000 --batch-size 8 --accum 2 --eval-every 500
.\.venv\Scripts\python.exe prepare_expanded_data.py
.\.venv\Scripts\python.exe train_indic.py --steps 12000 --init-checkpoint adivyanta_indic\best.pt --data-dir data\indic_expanded --out-dir adivyanta_indic\expanded --batch-size 8 --accum 2 --lr 0.0001
.\.venv\Scripts\python.exe fine_tune_indic_chat.py --base-checkpoint adivyanta_indic\expanded\best.pt --data-dir data\indic_expanded --out-dir adivyanta_indic\expanded --steps 3000
.\.venv\Scripts\python.exe evaluate_indic.py --checkpoint adivyanta_indic\expanded\chat_best.pt --data-dir data\indic_expanded
.\.venv\Scripts\python.exe benchmark_indic.py --checkpoint adivyanta_indic\expanded\chat_best.pt --out benchmarks\indic_expanded_outputs.jsonl
.\.venv\Scripts\python.exe prepare_chatmix_v2.py
.\.venv\Scripts\python.exe validate_chatmix_v2.py
.\.venv\Scripts\python.exe train_indic.py --steps 17000 --init-checkpoint adivyanta_indic\expanded\chat_best.pt --data-dir data\chatmix_v2 --out-dir adivyanta_indic\chatmix_v2_refined --batch-size 8 --accum 2 --eval-every 500 --lr 0.00015
.\.venv\Scripts\python.exe benchmark_indic.py --checkpoint adivyanta_indic\chatmix_v2_refined\best.pt --out benchmarks\indic_chatmix_v2_outputs.jsonl
```

The data preparation steps download the stated **datasets**, not model weights. They need several gigabytes of free disk space. Training used an RTX 2050 with 4 GB VRAM. `adivyanta_indic/best.pt` is the first corpus checkpoint created locally when reproducing training; `adivyanta_indic/expanded/best.pt` continues those scratch weights on the larger corpus; `expanded/chat_best.pt` is its conversational continuation. The GitHub repo ships the two expanded checkpoints needed for chat. The first-phase weights and full optimizer checkpoint `last.pt` stay local to keep the repo size manageable.

## Results and limits

Training and test metrics are published in `adivyanta_indic/` with the checkpoints. [The benchmark script](benchmark_indic.py) records raw model-only answers to fixed English, Hindi, Hinglish, and roast prompts under `benchmarks/`; [the math script](benchmark_math.py) checks exact answers on a fixed held-out GSM8K subset. A low response-token loss does not prove a useful or funny answer. The test split shares source datasets with training and measures fit to those sources; the fixed prompts provide a small separate behavioral check. The released model should be judged from its actual replies, especially on novel Hindi/Hinglish prompts and sensitive situations. **There is no measured evidence that this 46M model is better than Qwen or another established general assistant.**

[The measured benchmark report](benchmarks/REPORT.md) records the gains and regressions. On a fixed 50-question held-out GSM8K subset, the final neural model scored **0/50 exact answers**. Its raw English, Hindi, and roast responses are often off-topic or malformed. This is an honest experimental checkpoint; the app's memory and calculator cannot be counted as proof that the model itself reasons well.

The 9.71M legacy checkpoint in `checkpoints/` and its roast continuation in `adivyanta_10m/` were trained on [Meta EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues), which is **CC BY-NC 4.0**. They are noncommercial with attribution. They were **not** used to initialize the new Indic GPT. [LICENSE-CODE](LICENSE-CODE) grants MIT terms to the original code and the new scratch Indic weights, with the legacy weights explicitly excluded.
