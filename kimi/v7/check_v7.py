"""Validity checks on v7/v7_train.jsonl before the freeze (same bar as v6's finalize): no val problem or copy of one, nothing excluded, no Claude-written
target, no JAXBench task name, every exam-scale target byte-identical to the code the TPU judged with a pass/correct_slow verdict, and token lengths
within train_lora's 24,576 and the eval's 8,192-token answer cap (Gemma tokenizer, thinking off, as train_lora encodes)."""
import collections, glob, json, pathlib, re, statistics
V7 = pathlib.Path(__file__).resolve().parent; W = pathlib.Path.home() / "Projects/gemma-work"
import sys
FILE = sys.argv[1] if len(sys.argv) > 1 else "v7_train.jsonl"; BASE = sys.argv[2] if len(sys.argv) > 2 else "v6_train.jsonl"
rows = [json.loads(l) for l in open(V7 / FILE)]
fails = collections.defaultdict(list)
# val protection
val = [json.loads(l) for l in open(W / "data/v4_val.jsonl")]
vt = {v["task"] for v in val}
stems = {t.replace("__", "/", 2) for t in vt} | vt
for r in rows:
    t = r["task"]
    if t in vt or any(t.startswith(v + "__") for v in vt) or re.sub(r"(__[a-z0-9_]+)+$", "", t) in vt:
        fails["val_family"].append(r["problem_id"])
# exclusions (v6 lists) incl. via scaled_from
L = lambda f: {l.strip() for l in open(W / "wt-kimi/kimi/v6" / f) if l.strip() and not l.startswith("#")}
X = L("exclude_hard.txt") | L("flag_unsure.txt")
for r in rows:
    if r["problem_id"] in X or (r.get("scaled_from") or "") in X:
        fails["excluded_problem"].append(r["problem_id"])
# no Claude-written targets; no JAXBench names
jax = {json.loads(l)["task"] for l in open("/Users/hrudaynara/Projects/gemma-hackathon-research/control/runs/gemma4_e4b-it-qat__modal/bare.jsonl")}
for r in rows:
    if r.get("teacher") not in ("kimi-k3.1", "generator-v7"):
        fails["teacher"].append((r["problem_id"], r.get("teacher")))
    if r["task"] in jax:
        fails["jaxbench_name"].append(r["task"])
v4only = {json.loads(l)["id"].split("/", 1)[1] for l in open(V7 / "kn_items_v4only.jsonl")}
fails["v4only_claude"] = [r["problem_id"] for r in rows if r["problem_id"] in v4only]
# exam rows: target == TPU-judged code, verdict pass/correct_slow
halt = lambda x: x.get("verdict") == "harness_oom" and any(h in (x.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))
judged = {}
for f in glob.glob(str(W / "tpu/ops1/repo/oracle/jobs/kimi-kn*/results/*.jsonl")) + glob.glob(str(W / "tpu/ops1/repo/oracle/jobs/kimi-r3-*/results/*.jsonl")):
    for l in open(f):
        if l.strip() and '"meta": true' not in l:
            x = json.loads(l)
            if not halt(x):
                judged[x["id"]] = x
code = {}
for f in glob.glob(str(W / "tpu/ops1/kimi-kn*.items.jsonl")) + glob.glob(str(W / "tpu/fleet/kimi-kn*-s*.items.jsonl")) + glob.glob(str(W / "tpu/fleet/kimi-r3-s*.items.jsonl")):
    for l in open(f):
        x = json.loads(l); code[x["id"]] = x["code"].rstrip("\n")
PFX = {"e4_kn": "kimiKN0/", "e4_fix": "kimiKN1/", "e4_fix2": "kimiKN2/", "r3_gap": "kimiR3/"}
unfence = lambda c: c[len("```python\n"):-len("\n```")] if c.startswith("```python\n") and c.endswith("\n```") else c
for r in rows:
    if r.get("set") not in PFX:
        continue
    jid = PFX[r["set"]] + r["problem_id"]
    v = judged.get(jid)
    if not v or v["verdict"] not in ("pass", "correct_slow"):
        fails["exam_not_verified"].append(jid)
    if unfence(r["messages"][-1]["content"]).rstrip("\n") != code.get(jid):
        fails["exam_code_mismatch"].append(jid)
# base rows: byte-identical to rows of the v6 A6 file; every row is a base row, a TPU-verified added row or a generator row
import sys as _s; _s.path.insert(0, "/Users/hrudaynara/Projects/gemma-hackathon-research/control"); import bench as _b
def _eval_form(r):
    m = [dict(x) for x in r["messages"]]
    for i in range(2, len(m) - 1):
        if m[i]["role"] == "assistant":
            c = _b.extract_code(m[i]["content"])
            if c: m[i]["content"] = "```python\n" + c.rstrip("\n") + "\n" + "\n```"  # = what bench.py re-sends for a normally fenced answer
    t = _b.extract_code(m[-1]["content"])
    if t: m[-1]["content"] = "```python\n" + t.rstrip("\n") + "\n```"
    return {**r, "messages": m}
v6rows = {json.dumps(_eval_form(json.loads(l))) for l in open(W / "wt-kimi/kimi/v6" / BASE) if l.strip()}  # v6 rows as rendered in eval form
# eval form of a history turn: "```python\n" + extract_code(answer) + "\n```", i.e. the code's own trailing newline plus the wrapper's: exactly "\n\n```" at the end
fails["history_not_eval_form"] = [r["example_id"] for r in rows if any(m["role"] == "assistant" and not (m["content"].startswith("```python\n") and m["content"].endswith("\n\n```") and not m["content"].endswith("\n\n\n```")) for m in r["messages"][2:-1])]
fails["target_not_single_newline"] = [r["example_id"] for r in rows if not r["messages"][-1]["content"].endswith("\n```") or r["messages"][-1]["content"].endswith("\n\n```")]
for r in rows:
    if r.get("set") not in PFX and r.get("set") != "e5_gen" and json.dumps(r) not in v6rows:
        fails["base_row_not_in_v6"].append(r["example_id"])
ids = [r["example_id"] for r in rows]
fails["duplicate_example_id"] = [i for i, n in collections.Counter(ids).items() if n > 1]
devg = {json.loads(l)["example_id"] for l in open(V7 / "dev_items_gen.jsonl")}
r2 = set(json.load(open(V7 / "a6r2/drops.json"))["drops"]); r2g = set(json.load(open(V7 / "a6r2/aarav_drops.json"))["drop_generated_training_example_ids"])
r2b = set(json.load(open(V7 / "a6r2/base_drops.json"))["drops"])
fails["a6_rule2_base_row"] = [r["example_id"] for r in rows if r["problem_id"] in r2b]
fails["a6_rule2_added_row"] = [r["example_id"] for r in rows if (r.get("set") in PFX and r["problem_id"] in r2) or (r.get("set") == "e5_gen" and r["example_id"] in r2g)]
fails["dev_item_in_train"] = [r["example_id"] for r in rows if r["example_id"] in devg or r.get("surface_of") in devg]  # surface_of: a surface_gen.py variant
r2g = set(json.load(open(V7 / "a6r2/aarav_drops.json"))["drop_generated_training_example_ids"]) if (V7 / "a6r2/aarav_drops.json").exists() else set()
fails["a6r2_generated_in_train"] = [r["example_id"] for r in rows if r["example_id"] in r2g or r.get("surface_of") in r2g]
fails["thinking_not_false"] = [r["example_id"] for r in rows if r.get("thinking") is not False]
VAL = V7 / FILE.replace(".jsonl", "_val.jsonl")
if VAL.exists():
    vr = [json.loads(l) for l in open(VAL)]
    fails["val_count_not_29"] = [] if len(vr) == len(devg) else [len(vr)]
    fails["val_task_in_train"] = [v["task"] for v in vr if v["task"] in {r["task"] for r in rows}]
    fails["val_thinking_not_false"] = [v["example_id"] for v in vr if v.get("thinking") is not False]
# format: last turn is the fenced target; system prompt identical across rows
sys0 = rows[0]["messages"][0]
for r in rows:
    if r["messages"][0] != sys0:
        fails["system_prompt_differs"].append(r["example_id"])
    if not r["messages"][-1]["content"].startswith("```python\n") or r["messages"][-1]["role"] != "assistant":
        fails["target_not_fenced_code"].append(r["example_id"])
# tokens
from transformers import AutoTokenizer  # noqa: E402
TOK = str(next(pathlib.Path.home().glob(".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/*")))
tok = AutoTokenizer.from_pretrained(TOK)
A, T = [], []
for r in rows:
    m = r["messages"]
    p = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(m, tokenize=False, enable_thinking=False).rstrip("\n")
    assert full.startswith(p)
    a = len(tok(full[len(p):], add_special_tokens=False).input_ids); pl = len(tok(p, add_special_tokens=False).input_ids)
    A.append(a); T.append(a + pl)
fails["answer_over_8192"] = [i for i, a in enumerate(A) if a > 8192]
fails["total_over_24576"] = [i for i, t in enumerate(T) if t > 24576]
print(f"{len(rows)} rows | answer tokens median {statistics.median(A):.0f} max {max(A)} | total median {statistics.median(T):.0f} max {max(T)}")
bad = {k: v for k, v in fails.items() if v}
print("ALL CHECKS PASS" if not bad else "FAILURES: " + json.dumps({k: (len(v), v[:3]) for k, v in bad.items()}))
