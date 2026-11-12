"""Print the generated code (and optionally the rendered prompt) for given row ids."""
import json
import sys

run, ids = sys.argv[1], sys.argv[2:]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{run}/bare.jsonl")}
for i in ids:
    r = rows[i]
    print(f"===== {i} tokens={r['output_tokens']} prompt_tokens={r['prompt_tokens']}")
    print(r["code"])
