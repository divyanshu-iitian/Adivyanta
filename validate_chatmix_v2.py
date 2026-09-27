"""Check published ChatMix split integrity and generated arithmetic answers."""
import json
from pathlib import Path
import re


ROOT = Path(__file__).parent / "data" / "chatmix_v2"
EXPR = re.compile(r"(\d+)\s*([+*-])\s*(\d+)")


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    seen = set()
    math = 0
    for split, expected in manifest["splits"].items():
        count = 0
        with (ROOT / f"{split}.jsonl").open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                key = " ".join(row["user"].split()).casefold()
                if key in seen:
                    raise ValueError(f"Duplicate prompt across rows: {key[:80]}")
                seen.add(key)
                if row["source_license"] != manifest["license_by_source"][row["source"]]:
                    raise ValueError(f"Wrong license for {row['source']}")
                if row["source"] == "generated_math":
                    match = EXPR.search(row["user"])
                    if not match:
                        raise ValueError("Unparseable generated arithmetic prompt")
                    a, op, b = match.groups()
                    a, b = int(a), int(b)
                    value = a + b if op == "+" else a - b if op == "-" else a * b
                    if int(re.search(r"-?\d+", row["assistant"]).group()) != value:
                        raise ValueError("Incorrect generated arithmetic answer")
                    math += 1
                count += 1
        if count != expected:
            raise ValueError(f"Count mismatch for {split}: {count} != {expected}")
    print(json.dumps({"splits": manifest["splits"], "unique_prompts": len(seen),
                      "verified_math": math, "license_mismatches": 0}))


if __name__ == "__main__":
    main()
