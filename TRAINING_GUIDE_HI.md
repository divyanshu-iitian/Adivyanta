# Adivyanta ko scratch se kaise train kiya

## 1. Data jama aur saaf kiya

`prepare_indic_data.py` licensed Aya, OpenAssistant aur Hinglish datasets ke selected prompt/reply pairs download karta hai. `prepare_expanded_data.py` extra Hindi Aya, UltraChat aur GSM8K examples add karta hai. Source versions aur counts [DATASETS.md](DATASETS.md) aur model manifests mein hain. Prompt ke hash se train, validation aur test alag hote hain; same normalized prompt do splits mein nahi jana chahiye. Raw prepared JSONL files `data/` mein local rehti hain.

## 2. Apna tokenizer banaya

16,384-token byte-level BPE tokenizer **sirf training split** se seekhta hai. Tokenizer text ko token IDs mein badalta hai. Kisi existing GPT/Qwen ka tokenizer ya weight load nahi hota. Hindi, Roman Hindi aur English sab isi vocabulary ko share karte hain.

## 3. Random weights se GPT start kiya

`model.py` ka decoder-only Transformer 12 layers, 8 attention heads, 512 hidden dimensions, 256-token context aur tied input/output embeddings use karta hai. Total **46,349,312 parameters** hain. Start mein weights random the. Har example mein model ko user prompt diya jata hai aur assistant reply ka agla token predict karaya jata hai. Prompt tokens ka loss mask hota hai, reply tokens ka cross-entropy loss train hota hai.

## 4. Do training phases

Pehla phase `train_indic.py` se original corpus par 6,000 optimizer steps tha. Doosra phase usi scratch checkpoint ko expanded corpus par 12,000 total steps tak continue karta hai. Ek step mein batch size 8 aur gradient accumulation 2 se 16 examples process hote hain. AdamW, mixed precision aur gradient clipping use hote hain. Har 500 steps validation loss nikalta hai; sabse achha checkpoint `best.pt` mein save hota hai. `last.pt` optimizer ke saath local resume ke liye hai.

`fine_tune_indic_chat.py` conversational examples par ek aur continuation karta hai. Yeh bhi scratch weights hi hain; koi doosra model ismein merge nahi hota. Iska `chat_best.pt` alag chat-validation slice se select hota hai.

## 5. Result kaise check kiya

`evaluate_indic.py` held-out test par language-wise response-token loss/perplexity likhta hai. `benchmark_indic.py` English, Hindi, Hinglish aur roast prompts ke **raw model-only** answers save karta hai. `benchmark_math.py` official GSM8K test se seeded 50 questions par exact-number accuracy check karta hai. Chat app ki memory, calculator aur authored roast/opening replies in model-only benchmarks mein nahi chalti. Loss kam hona useful ya truthful response ki guarantee nahi hai.

## 6. Chat memory kaise kaam karti hai

`conversation_memory.py` explicit facts, recent turns aur `/correct` feedback ko local JSON mein rakhta hai. Jab tum `my name is Divyanshu` kehte ho, naam directly save hota hai; next session mein bhi milta hai. Har chat message se neural weights automatically update nahi hote. `train_feedback.py` sirf saved corrections par optional local update karta hai aur validation gate pass karne par personalized checkpoint save karta hai. Memory aur personalized checkpoint GitHub par upload nahi hote.

## Khud reproduce karne ke commands

Windows PowerShell mein, repo ke andar `.venv` aur CUDA PyTorch set up karne ke baad:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe prepare_indic_data.py
.\.venv\Scripts\python.exe train_indic.py --steps 6000 --batch-size 8 --accum 2
.\.venv\Scripts\python.exe prepare_expanded_data.py
.\.venv\Scripts\python.exe train_indic.py --steps 12000 --init-checkpoint adivyanta_indic\best.pt --data-dir data\indic_expanded --out-dir adivyanta_indic\expanded --batch-size 8 --accum 2 --lr 0.0001
.\.venv\Scripts\python.exe fine_tune_indic_chat.py --base-checkpoint adivyanta_indic\expanded\best.pt --data-dir data\indic_expanded --out-dir adivyanta_indic\expanded --steps 3000
.\.venv\Scripts\python.exe chat.py
```

Checkpoint aur raw benchmark outputs dekhkar hi quality judge karo. Itne chhote model aur limited data/compute se Qwen jaise bade model ki general logic ya factual reliability ka claim sahi nahi hoga. Agar answer ka source verified na ho, model galat bhi bol sakta hai.
