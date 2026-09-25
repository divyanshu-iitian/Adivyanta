"""Fixed, model-only probes for the scratch multilingual checkpoint."""
import json
import argparse
from pathlib import Path
import sys

import torch

from conversation_memory import Memory
from smart_chat import generate, load_model


ROOT = Path(__file__).parent
PROMPTS = [
    ("english", "Hey, I had a long day. What should we talk about?"),
    ("english", "I keep forgetting where I put my keys. Any ideas?"),
    ("english", "Explain what a compiler does in one sentence."),
    ("english", "Tell me something fun about street food in India."),
    ("hinglish", "Kya scene hai, aaj kuch interesting batao."),
    ("hinglish", "Mujhe padhai mein focus nahi ho raha, kya karun?"),
    ("hinglish", "Golgappe aur samose mein tum kya choose karoge?"),
    ("hinglish", "Mera code baar baar crash ho raha hai, help karo."),
    ("hinglish", "Thoda funny sa joke suna do."),
    ("hinglish", "Mujhe ek naye dost se baat shuru karni hai."),
    ("hindi", "आज का दिन कैसा रहा?"),
    ("hindi", "मुझे पढ़ाई के लिए एक छोटा सुझाव दो।"),
    ("hindi", "समोसे के बारे में एक मज़ेदार बात कहो।"),
    ("hindi", "कंप्यूटर प्रोग्राम क्या होता है?"),
    ("roast", "Roast my habit of losing my umbrella every week."),
    ("roast", "Meri bahut lambi to-do list ka roast karo."),
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "indic_model_outputs.jsonl")
    args = parser.parse_args()
    model, tok = load_model(args.checkpoint, prefer_local=False)
    memory = Memory(ROOT / "data" / "indic_benchmark_empty_memory.json")
    memory.data = {"facts": {}, "notes": [], "turns": [], "corrections": {}}
    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for index, (language, prompt) in enumerate(PROMPTS, 1):
            torch.manual_seed(1000 + index)
            response = generate(model, tok, memory, prompt)
            record = {"id": index, "language": language, "prompt": prompt, "model_only_response": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
