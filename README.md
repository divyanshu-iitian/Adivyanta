# Adivyanta

A small, locally runnable chat and playful roast project. It contains a **9.71M parameter GPT trained from random weights** and a stronger option: a **4.01M trainable parameter LoRA adapter** for [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct). The 360M base is Hugging Face's model; Adivyanta did **not** train it from scratch. The default app uses the adapter for uncovered roast topics, authored lines for recognized topics, and the base model for ordinary chat.

**What to expect:** The scratch 10M model often produces incoherent replies. The 360M adapter is more direct on roast requests, but can produce weak or repetitive jokes on unseen topics. Adivyanta is **not shown to be generally better** than SmolLM2 or Qwen. It is not sentient; next-token training does not establish consciousness.

## Run chat on Windows PowerShell

Python 3.11+ is recommended. Install a suitable [PyTorch build](https://pytorch.org/get-started/locally/) first:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe chat.py
```

The first launch downloads the approximately 360M parameter SmolLM2 base from Hugging Face. An internet connection is needed then. Later launches use the local cache. CPU loading and generation are slower. Type `/quit` to exit. For one reply:

```powershell
.\.venv\Scripts\python.exe chat.py --prompt "Roast my 50 browser tabs."
```

`chat.py` now starts Adivyanta by default. The educational scratch GPT is available with `chat.py --scratch`; its roast fine-tuned checkpoint with `chat.py --checkpoint adivyanta_10m/best.pt`. Use `chat_adivyanta.py --model-only --prompt "..."` to bypass authored lines and routing, or `--base-only` for original SmolLM2. The adapter requires that base. `KeyboardInterrupt` during an import means Python was interrupted, usually by Ctrl+C; it does not mean `tokenizers` failed to install. Verify with `.\.venv\Scripts\python.exe -c "from tokenizers import Tokenizer; print('OK')"`.

## What was trained

| Component | Starting point | Trained weights | Data | Best held-out response-token loss |
| --- | --- | ---: | --- | ---: |
| Original GPT | Random initialization | 9,708,672 parameters | Meta EmpatheticDialogues | 3.5011 on 64 validation examples; 3.6750 on 8,401 test examples |
| Adivyanta 10M | Original GPT checkpoint | 9,708,672 parameters continued | 292 original roast/control examples | 5.2365 before to **4.4431** at step 50 |
| Adivyanta 360M adapter | SmolLM2-360M-Instruct | 4,014,080 LoRA parameters | The same 292 original examples | 4.8420 before to **3.7423** at epoch 3 |

The 10M and 360M losses use different tokenizers and are **not directly comparable**. The roast validation set has 36 replies and shares topics with training, so its loss mainly tests style learning. Later 10M steps and the adapter's fourth epoch worsened validation loss; saved checkpoints are the best earlier ones. See [scratch metrics](checkpoints/metrics.json), [scratch test results](checkpoints/evaluation.json), [10M continuation metrics](adivyanta_10m/training_metrics.json), and [adapter metrics](adivyanta_adapter/training_metrics.json).

### Training data and rights

The scratch GPT uses [Meta EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues), whose dataset card lists **CC BY-NC 4.0**. Its weights and continued 10M weights should be used noncommercially with attribution. The additional roast examples in [roast_data.py](roast_data.py) were written for this project. The adapter's base model is [Apache 2.0](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct). No live Reddit posts were collected: [Reddit's API terms](https://redditinc.com/policies/data-api-terms) do not grant general AI training rights for user content.

To rebuild the scratch dataset, obtain Meta's original archive and run:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest 'https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz' -OutFile 'data/empatheticdialogues.tar.gz'
tar -xzf data/empatheticdialogues.tar.gz -C data
.\.venv\Scripts\python.exe prepare_data.py
```

`prepare_data.py` retains Meta's train/validation/test conversation split, converts adjacent turns to prompt/reply examples, and fits a 2,048-token byte-level BPE tokenizer on training data only. There are 58,829 training examples from 17,780 conversations. The GPT has five causal Transformer layers, six attention heads, 384 hidden dimensions, a 128-token context, and tied input/output embeddings. `train.py` uses response-token cross-entropy, AdamW, gradient clipping, and a warmup/cosine schedule. It ran to 10,000 steps on an RTX 2050. The complete untouched test split had 8,401 examples and perplexity 39.45.

To reproduce the roast data and continue training:

```powershell
.\.venv\Scripts\python.exe roast_data.py
.\.venv\Scripts\python.exe fine_tune_10m.py
.\.venv\Scripts\python.exe fine_tune_360.py --epochs 4
```

`fine_tune_10m.py` continues all scratch GPT weights. `fine_tune_360.py` freezes the SmolLM2 base and trains LoRA matrices in attention and feed-forward projections, predicting only reply tokens. The exact base revision is `a10cc1512eabd3dde888204e902eca88bddb4951`. Four epochs were measured; the best epoch 3 adapter was saved. This is **fine-tuning**, not creating a 360M foundation model. Neither model acquires awareness.

## Benchmark and comparison

The [40 prompt set and raw outputs](benchmarks/raw_outputs.jsonl) contain 36 roast prompts and four ordinary or sensitive controls. The [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) comparison uses the same prompts; its [raw outputs](benchmarks/qwen_outputs.jsonl) are separate. [The score script](score_benchmark.py) reports basic format indicators on roast prompts:

| Model | One line, 30 words or fewer | Starts by addressing user | Disclaimer phrase |
| --- | ---: | ---: | ---: |
| Original scratch 10M | 36/36 | 5/36 | 0/36 |
| Adivyanta continued 10M | 36/36 | 35/36 | 0/36 |
| SmolLM2-360M base | 33/36 | 3/36 | 19/36 |
| Adivyanta 360M adapter only | 36/36 | 33/36 | 1/36 |
| Qwen2.5-0.5B-Instruct | 33/36 | 0/36 | 0/36 |

These indicators measure **format and style only**. A direct opening is not necessarily a good joke. For example, the adapter replies to a karaoke roast with `Karaoke kya hai, kya hai?`; its response to `My mother is sick` is unsuitable. The original and continued 10M replies often contain nonsense words. Qwen also gave unsuitable replies to some controls in this small local run. The prompt set is small, mostly authored alongside the training examples, and has no blind human humor ratings. It cannot support a claim of overall superiority.

The [routed app outputs](benchmarks/routed_outputs.jsonl) record `curated`, `adapter`, `base`, or `redirect` for every prompt. Authored lines cover known topics; model-only outputs are in the raw benchmark. This distinction matters because authored replies can improve the app without improving model weights. Run the benchmarks again with:

```powershell
.\.venv\Scripts\python.exe benchmark_adivyanta.py
.\.venv\Scripts\python.exe benchmark_qwen.py
.\.venv\Scripts\python.exe score_benchmark.py
.\.venv\Scripts\python.exe benchmark_routed.py
```

Generation can vary by software, hardware, and revision; inspect raw outputs as well as counts. The strongest supported claim is narrow: **the adapter follows the requested one-line roast style more often than the unfine-tuned SmolLM2 base on this 36-prompt set, and lowers response-token loss on its related validation set**. Its humor quality, general chat, and safety have not been established as better than existing models.

## How the chat app works

`chat.py` loads the 360M base and adapter. A roast about a recognized habit or hobby gets an authored line from [curated_roast.py](curated_roast.py); an unrecognized roast goes to the adapter; ordinary chat uses the unadapted base; identity-based roast requests get a redirect. This routing is imperfect. The raw adapter should not be trusted for distress or other sensitive situations. The scratch GPT has a shorter context and poor quality; it shows every step of training from random weights.
