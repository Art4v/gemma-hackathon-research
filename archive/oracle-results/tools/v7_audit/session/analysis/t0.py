"""Per-task turn-0 verdicts for a JAXBench run, sorted by task number."""
import json
import re
import sys

run = sys.argv[1]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{run}/bare.jsonl")}
verd = {}
for l in open(f"{run}/verdicts.jsonl"):
    v = json.loads(l)
    verd[v["id"]] = v  # later lines (TPU) overwrite CPU step-1 lines


def num(task):
    return int(re.match(r"\d+", task).group())


for vid in sorted((i for i in verd if i.endswith("/t0")), key=lambda i: num(verd[i]["task"])):
    v, r = verd[vid], rows.get(vid, {})
    msg = (v.get("msg") or "").replace("\n", " ")
    print(f"{v['task'][:34]:34s} {v['verdict']:13s} {str(v.get('failed_stage'))[:8]:8s} tok={r.get('output_tokens','?'):>5} fin={r.get('finish','?'):6s} | {msg[:150]}")
