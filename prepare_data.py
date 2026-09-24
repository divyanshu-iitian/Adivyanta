"""Prepare Meta's CC BY-NC 4.0 EmpatheticDialogues as chat examples."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).parent
SOURCE = ROOT / "data" / "empatheticdialogues"
OUT = ROOT / "data" / "processed"
SPECIAL = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|unk|>"]


def clean(s):
    return " ".join(s.replace("_comma_", ",").replace("\n", " ").split())


def conversations(split):
    groups = defaultdict(list)
    with (SOURCE / f"{split}.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            utterance = clean(row["utterance"])
            if utterance:
                groups[row["conv_id"]].append((int(row["utterance_idx"]), utterance))
    for turns in groups.values():
        turns.sort()
        if len(turns) > 1:
            yield [text for _, text in turns]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_convs = list(conversations("train"))
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator((s for conv in train_convs for s in conv), trainer=trainers.BpeTrainer(vocab_size=2048, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()), length=sum(map(len, train_convs)))
    tok.save(str(OUT / "tokenizer.json"))
    stats = {}
    for split, convs in (("train", train_convs), ("valid", list(conversations("valid"))), ("test", list(conversations("test")))):
        count = 0
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for conv in convs:
                for end in range(1, len(conv)):
                    history = conv[max(0, end - 3):end]
                    prefix = "<|bos|>" + "".join(("<|user|>" if i % 2 == (len(history) - 1) % 2 else "<|assistant|>") + text for i, text in enumerate(history)) + "<|assistant|>"
                    response = conv[end]
                    # Store text to keep the dataset easy to inspect and re-tokenize.
                    f.write(json.dumps({"prefix": prefix, "response": response}, ensure_ascii=False) + "\n")
                    count += 1
        stats[split] = {"conversations": len(convs), "examples": count}
    stats["vocab_size"] = tok.get_vocab_size()
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
