"""Adivyanta's scratch-trained GPT chat with inspectable local memory."""
import argparse
from pathlib import Path
import re
import sys

import torch
from tokenizers import Tokenizer

from conversation_memory import DEFAULT_PATH, Memory
from curated_roast import choose
from grounded_tools import arithmetic_answer
from model import GPT, GPTConfig


ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "adivyanta_indic" / "best.pt"
MODEL_CHOICES = (
    ROOT / "adivyanta_indic" / "chatmix_v2_refined" / "best.pt",
    ROOT / "adivyanta_indic" / "expanded" / "chat_best.pt",
    ROOT / "adivyanta_indic" / "expanded" / "best.pt",
    ROOT / "adivyanta_indic" / "chat_best.pt",
    CHECKPOINT,
)
ROAST_RE = re.compile(r"\b(roast\w*|make fun|joke about|mazaak|mazak|leg.pull|savage)\b", re.I)
NO_ROAST_RE = re.compile(r"\b(?:don't|do not|no)\s+(?:(?:want|need)\s+)?(?:a\s+)?roast\b", re.I)
SENSITIVE_RE = re.compile(r"\b(religion|religious|caste|race|ethnicity|disability|illness|disease|death)\b", re.I)
HINGLISH_RE = re.compile(r"\b(?:kya|mera|meri|mujhe|tum|tumhe|kaise|kar|rahe|hai|hain|yaar|bhai|batao|karo|nahi|accha|achha|aaj|kal)\b", re.I)


def familiar_reply(message, memory):
    """Reliable replies to a few common chat openers; separate from model output."""
    clean = re.sub(r"[!?.\s]+", " ", message.casefold()).strip()
    name = memory.data["facts"].get("name")
    if clean in {"hi", "hello", "hey", "namaste", "नमस्ते"}:
        return f"Namaste{', ' + name if name else ''}! Aaj kya baat karein?"
    if clean in {"kya kar rahe ho", "kya kar rhe ho", "क्या कर रहे हो"}:
        return "Tumse baat kar raha hoon. Batao, aaj kya chal raha hai?"
    if clean in {"samosa", "समोसा"}:
        return "Samosa! Hari chutney ke saath ya imli wali?"
    if clean in {"what is your name", "what's your name", "whats your name", "your name", "tumhara naam kya hai", "aapka naam kya hai", "तुम्हारा नाम क्या है", "आपका नाम क्या है"}:
        return "Mera naam Adivyanta hai."
    if clean in {"i love you", "love you", "mujhe tumse pyaar hai", "मुझे तुमसे प्यार है"}:
        return "Yeh sunkar achha laga! Main Adivyanta hoon; tumse baat karne ke liye yahan hoon."
    if clean in {"ha", "haan", "han", "yes", "yep", "ji"}:
        return "Haan, batao—kis baare mein baat karein?"
    return None


def is_hinglish(message):
    return bool(HINGLISH_RE.search(message)) and not bool(re.search(r"[\u0900-\u097f]", message))


def load_model(checkpoint=None, prefer_local=True):
    if checkpoint is None:
        checkpoint = next((path for path in MODEL_CHOICES if path.exists()), CHECKPOINT)
        personalized = ROOT / "data" / "personalized.pt"
        if prefer_local and personalized.exists():
            checkpoint = personalized
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Scratch model not trained yet: {checkpoint}. Run prepare_indic_data.py and train_indic.py.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = checkpoint.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = ROOT / "adivyanta_indic" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tokenizer_path))
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if set(missing) - {"head.weight"} or unexpected:
        raise ValueError(f"Invalid checkpoint keys: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, tok


@torch.inference_mode()
def generate(model, tok, memory, message, max_new_tokens=80, temperature=0.75, top_k=30, greedy=False):
    history = memory.data["turns"][-4:]
    pieces = ["<|bos|>"]
    for turn in history:
        pieces.append(("<|user|>" if turn["role"] == "user" else "<|assistant|>") + turn["content"])
    pieces.append("<|user|>" + message + "<|assistant|>")
    ids = tok.encode("".join(pieces), add_special_tokens=False).ids
    ids = ids[-(model.cfg.block_size - max_new_tokens):]
    prompt_len = len(ids)
    stop = {tok.token_to_id(s) for s in ("<|eos|>", "<|user|>", "<|assistant|>", "<|pad|>")}
    device = next(model.parameters()).device
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.cfg.block_size:]], device=device)
        logits, _ = model(x)
        scores = logits[0, -1].float() / temperature
        scores[tok.token_to_id("<|pad|>")] = -float("inf")
        for previous in set(ids[prompt_len:]):
            if scores[previous] > 0:
                scores[previous] /= 1.2
            else:
                scores[previous] *= 1.2
        threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores[scores < threshold] = -float("inf")
        next_id = scores.argmax().item() if greedy else torch.multinomial(torch.softmax(scores, dim=-1), 1).item()
        if next_id in stop:
            break
        ids.append(next_id)
    return tok.decode(ids[prompt_len:], skip_special_tokens=True).strip()


def reply(model, tok, memory, message):
    changes = memory.learn(message)
    known = memory.known_answer(message)
    calculated = arithmetic_answer(message)
    familiar = familiar_reply(message, memory)
    if known:
        answer = known
    elif calculated is not None:
        answer = calculated
    elif familiar:
        answer = familiar
    elif re.search(r"\b(?:what do you remember|what do you know about me|tumhe mere bare mein kya yaad)\b", message, re.I):
        answer = memory.facts_text() or "Abhi koi personal fact save nahi hai. /remember se bata sakte ho."
    elif "name" in changes:
        answer = f"Theek hai, {memory.data['facts']['name']}! Naam yaad rakhunga."
    elif "girlfriend" in changes:
        answer = f"Samjha, {memory.data['facts']['girlfriend']} tumhari girlfriend hai."
    elif "creator" in changes:
        name = memory.data["facts"].get("name")
        answer = f"Haan{', ' + name if name else ''}, tumne Adivyanta project banaya hai. Kya improve karein?"
    elif "note" in changes:
        answer = "Yaad rakh liya. /memory se dekh sakte ho."
    elif ROAST_RE.search(message) and not NO_ROAST_RE.search(message) and SENSITIVE_RE.search(message):
        answer = "Kisi habit ya hobby ka roast karte hain. Topic batao, ek playful line dunga."
    elif ROAST_RE.search(message) and not NO_ROAST_RE.search(message) and choose(message)[0]:
        answer = choose(message)[0]
    else:
        answer = generate(model, tok, memory, message)
    memory.add_turn("user", message)
    memory.add_turn("assistant", answer)
    return answer


def command(memory, line):
    if line == "/memory":
        return (memory.facts_text() or "No saved facts.") + f"\nSaved turns: {len(memory.data['turns'])}; corrections: {len(memory.data['corrections'])}."
    if line.startswith("/remember "):
        memory.remember(line[len("/remember "):])
        return "Yaad rakh liya. /memory se dekh sakte ho."
    if line.startswith("/forget "):
        return "Bhool gaya." if memory.forget(line[len("/forget "):]) else "Woh memory key nahi mili. Try name, girlfriend, language, creator, likes, notes, history, or all."
    if line.startswith("/correct "):
        body = line[len("/correct "):]
        if "=>" not in body:
            return "Use: /correct question => better answer"
        question, answer = body.split("=>", 1)
        memory.correct(question, answer)
        return "Correction saved. Isi question par agle baar ye answer dunga."
    if line == "/help":
        return "/memory, /remember fact, /forget name|girlfriend|language|creator|likes|notes|history|all, /correct question => answer, /quit"
    return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="One message and exit; memory still persists")
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    memory = Memory(args.memory_file)
    if args.prompt and args.prompt.startswith("/"):
        print(command(memory, args.prompt) or "Unknown command. Use /help.")
        return
    model, tok = load_model(args.checkpoint)
    alternate = None
    base_path = ROOT / "adivyanta_indic" / "expanded" / "best.pt"
    personalized = ROOT / "data" / "personalized.pt"
    v2_path = ROOT / "adivyanta_indic" / "chatmix_v2_refined" / "best.pt"
    if args.checkpoint is None and not personalized.exists() and not v2_path.exists() and base_path.exists() and (base_path.parent / "chat_best.pt").exists():
        alternate, _ = load_model(base_path, prefer_local=False)

    def answer_message(line):
        selected = model if alternate is None or is_hinglish(line) else alternate
        return reply(selected, tok, memory, line)

    if args.prompt:
        print(answer_message(args.prompt))
        return
    print("Adivyanta scratch GPT (English/Hindi/Hinglish, local memory). /help for commands; /quit to exit.")
    while True:
        try:
            line = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line == "/quit":
            break
        if not line:
            continue
        try:
            answer = command(memory, line) if line.startswith("/") else answer_message(line)
            print("Adivyanta:", answer or "Unknown command. Use /help.")
        except ValueError as exc:
            print("Adivyanta:", exc)


if __name__ == "__main__":
    main()
