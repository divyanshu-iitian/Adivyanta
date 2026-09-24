# Adivyanta 10M (experimental)

This is the original scratch-trained GPT continued on the Adivyanta roast examples. It has **9,708,672 parameters**. The base was trained on Meta EmpatheticDialogues (CC BY-NC 4.0); keep the source attribution and noncommercial restriction when using these weights.

On 36 held-out authored roast replies, response-token loss improved from **5.2365** to **4.4431** at fine-tune step 50. Later steps overfit and were discarded. The lower loss did **not** translate into good chat quality: the raw benchmark outputs include garbled words. This checkpoint is kept as a transparent before/after experiment, not the recommended Adivyanta chatbot.

Run with `python chat.py --checkpoint adivyanta_10m/best.pt`. The 2,048-token tokenizer is included here as `tokenizer.json`. The main `README.md` explains data preparation.
