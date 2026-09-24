"""Simple, reproducible format indicators; they do not measure how funny a roast is."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).parent
OUT = ROOT / "benchmarks"
rows = [json.loads(s) for s in (OUT / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
qwen_path = OUT / "qwen_outputs.jsonl"
if qwen_path.exists():
    qwen = {r["id"]: r for r in (json.loads(s) for s in qwen_path.read_text(encoding="utf-8").splitlines())}
    for row in rows:
        if row["id"] in qwen:
            row["qwen_05b"] = qwen[row["id"]]["qwen_05b"]

roasts = [r for r in rows if r["kind"] == "roast"]
model_names = ["original_10m", "adivyanta_10m", "base_360m", "adivyanta_360m"]
if all("qwen_05b" in r for r in roasts):
    model_names.append("qwen_05b")


def indicators(s):
    words = s.split()
    return {
        "nonempty": bool(s.strip()),
        "one_line_30_words_or_less": bool(s.strip()) and "\n" not in s and len(words) <= 30,
        "second_person_opening": bool(re.match(r"^(your|you|you're|teri|tere|tera|tu|aapka|aapki)\b", s.strip(), re.I)),
        "disclaimer_phrase": bool(re.search(r"\b(not sure|not equipped|as an ai|can't roast|cannot roast|i'm here to help|i can certainly|i'll just)\b", s, re.I)),
    }


report = {"roast_prompts": len(roasts), "control_prompts": len(rows) - len(roasts), "metrics": {}}
for model in model_names:
    by_row = [indicators(row[model]) for row in roasts]
    report["metrics"][model] = {name: sum(x[name] for x in by_row) for name in by_row[0]}
(OUT / "format_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
