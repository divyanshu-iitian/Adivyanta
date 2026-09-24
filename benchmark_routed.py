"""Evaluate the complete Adivyanta chat route and publish its source per reply."""
import json
from collections import Counter
from pathlib import Path

from chat_adivyanta import load_model, respond_routed, route_name


ROOT = Path(__file__).parent
rows = [json.loads(s) for s in (ROOT / "data" / "roast" / "benchmark_prompts.jsonl").read_text(encoding="utf-8").splitlines()]
model, tok = load_model()
out = ROOT / "benchmarks"
out.mkdir(exist_ok=True)
routes = Counter()
with (out / "routed_outputs.jsonl").open("w", encoding="utf-8") as f:
    for i, row in enumerate(rows, 1):
        route = route_name(row["prompt"])
        response = respond_routed(model, tok, [{"role": "user", "content": row["prompt"]}])
        result = {"id": i, "kind": row["kind"], "prompt": row["prompt"], "route": route, "response": response}
        routes[route] += 1
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)
(out / "routed_summary.json").write_text(json.dumps({"prompts": len(rows), "routes": dict(routes)}, indent=2), encoding="utf-8")
