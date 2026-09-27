"""Fixed novel prompts for comparing raw model weights without app replies."""
import argparse
import json
from pathlib import Path

import torch

from conversation_memory import Memory
from smart_chat import generate, load_model


ROOT = Path(__file__).parent
PROMPTS = [
    ("greeting", "Hey Adivyanta, kaise ho?"),
    ("relationship", "My girlfriend is Neha."),
    ("support", "I had a rough day and feel low. Can we talk?"),
    ("identity", "What kind of model are you?"),
    ("food", "Samosa ke baare mein ek fun baat batao."),
    ("food", "Burger ya samosa, kya choose karoge?"),
    ("knowledge", "Can you explain a compiler briefly?"),
    ("debugging", "Mera code baar baar crash hota hai. Kya check karun?"),
    ("study", "Padhai mein focus kaise badhaun?"),
    ("roast", "Roast my messy room in Hinglish."),
    ("identity", "Tum mere saath bahar chaloge?"),
    ("dating", "How do I politely ask someone on a date?"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    model, tok = load_model(args.checkpoint, prefer_local=False)
    memory = Memory(ROOT / "data" / "probe_memory_not_saved.json")
    memory.data = {"facts": {}, "notes": [], "likes": [], "turns": [], "corrections": {}, "state": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as target:
        for index, (intent, prompt) in enumerate(PROMPTS, 1):
            torch.manual_seed(42000 + index)
            response = generate(model, tok, memory, prompt, max_new_tokens=64, greedy=args.greedy)
            row = {"id": index, "intent": intent, "prompt": prompt, "model_only_response": response,
                   "decoding": "greedy" if args.greedy else "temperature_0.75_top_k_30"}
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
