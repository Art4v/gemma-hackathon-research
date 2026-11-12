"""Fallback v7 training file (Karthik's agent's proposal, #5 Nov 12 02:01 UTC), built only from verified inputs that need nothing new:
  v6_train.jsonl, the A6 file (hruday/kimi-k31 @ c8e64e3), minus Aarav's agent's 4 L3 drops
  + the scaled Kimi kernels that passed on the TPU unchanged (oracle-results/kimi-kn0-s1..4-p1), excluding Karthik's 16 v4 (Claude) kernels
  + the 269 generated rows (aarav/dataset-v4 dataset/v7_gen/v7_gen_rows.jsonl), minus a dev holdout of whole epilogue chains (about 10%)
Usage: python build.py <dir> <bench.py> ; writes train.jsonl, dev_items.jsonl, MANIFEST.json in <dir>.
"""
import collections, hashlib, importlib.util, json, pathlib, random, re, sys

D = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("bench", sys.argv[2]); bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
read = lambda p: [json.loads(l) for l in open(p) if l.strip()]
SC = D.parent

# 1. v6 strict minus the 4 L3 drops agreed on #5 (the within-set __scale pairing is kept)
DROP = {"kernelbook/Sentence_Maxpool_14501", "kernelbook/MetaBilinear_711", "kernelbook/CPC_15255", "kernelbook/QMaxPooling2d_9618"}
v6 = read(SC / "modal5/data6/v6_train.jsonl")  # A6 adopted (all three sign-offs, #5 Nov 12 02:18 UTC)
base = [r for r in v6 if r["problem_id"] not in DROP]
assert len(v6) - len(base) == 4, len(v6) - len(base)

# 2. scaled Kimi kernels that passed unchanged at exam scale
verdict = {d["id"]: d for d in read(SC / "v7bb/kn0_rows.jsonl") if "verdict" in d}
claude16 = {x["id"] for x in read(D / "karthik16.jsonl")}
scaled = []
KN0_A6 = ("GateLayer_1320", "Maxout_4302")
for it in read(D / "kimi_items.jsonl"):
    v = verdict.get(it["id"])
    if it["id"] in claude16 or not v or v["verdict"] != "pass":
        continue
    if any(k in it["id"] for k in KN0_A6):  # A6 rule 2 per Hruday's agent's trace (#5 02:47 UTC); my re-trace missed these
        continue
    assert v["oracle_sha"] == "c9a07d9298a1"
    msgs = bench.conversation(it["ref_code"], [], None) + [{"role": "assistant", "content": f"```python\n{it['code']}\n```"}]
    scaled.append({"example_id": f"{it['task']}.kn0", "kind": "single", "messages": msgs, "verdict": "pass", "speed_vs_xla": float(v["speed_vs_xla"]),
                   "source": it["id"].split("/")[1], "task": it["task"], "problem_id": it["id"].split("/", 1)[1], "teacher": "kimi-k3.1",
                   "thinking": False, "rule3_family": False, "generic_stack": False, "set": "e6_kn0", "scaled_from": None,
                   "oracle_job": "kimi-kn0", "oracle_sha": "c9a07d9298a1"})

# 3. generated rows, minus a dev holdout of whole epilogue chains (the docstring names the chain; sizes and dtypes vary within a chain)
gen = read(D / "gen_rows.jsonl")
# A6 rule 2 (k-task chain equal once constants, scales and bias are set aside): aarav/dataset-v4 dataset/v7_gen/a6_rule2_drops.json
a6 = set(json.load(open(D / "a6_rule2_drops.json"))["drop_generated_training_example_ids"])
gen_all = gen
# dev split by NORMALISED chain (Karthik's agent's audit, #5 02:44 UTC): the reference's op_chain with constants, scales, bias and layout ops
# set aside, the same reading as the A6 rule-2 check, so no training row shares a dev item's normalised chain. A6 matches are in neither.
LIN = {"add", "sub", "mul", "div", "neg", "convert_element_type", "broadcast_in_dim", "reshape", "transpose", "squeeze", "expand_dims", "copy"}
SIG = {r["id"]: r["signature"]["op_chain"] for r in read(D / "gen_full_verified.jsonl") if r.get("ok")}
def chain(r):
    return r["family"].replace("bmm_epilogue", "matmul_epilogue") + ":" + " ".join(o for o in SIG[r["task"]] if o not in LIN)
pool = [r for r in gen if r["example_id"] not in a6]
by = collections.defaultdict(list)
for r in pool:
    by[chain(r)].append(r)
chains = sorted(by); random.Random("v7-dev").shuffle(chains)
chains.sort(key=lambda c: len(by[c]) > 4)  # whole small groups first, so 27 items cover many chains without emptying a large one
dev, dev_chains = [], []
for c in chains:
    if len(dev) >= 27:
        break
    dev += by[c]; dev_chains.append(c)
FIXED_DEV = D / "dev_fixed.jsonl"  # the shared dev set (#5, ca43b98): read, never recomputed, so it cannot shift when the pool changes
if FIXED_DEV.exists():
    want = [json.loads(l)["example_id"] for l in open(FIXED_DEV)]
    byid = {r["example_id"]: r for r in gen}
    dev = [byid[i] for i in want]; dev_chains = sorted({chain(r) for r in dev})
dev_ids = {r["example_id"] for r in dev}
gen_train = [r for r in pool if r["example_id"] not in dev_ids]
a6_not_passes = len(a6 - {r["example_id"] for r in gen})  # matched candidates that never passed the TPU: nothing to drop
assert not {chain(r) for r in gen_train} & set(dev_chains)

rows = base + scaled + gen_train
# Code turns as the eval renders them (Karthik's agent's corrected rule, #5 03:12 UTC). The model's reply ends with one newline before the
# fence; extract_code keeps it and conversation() adds one, so a history turn has two. Rule, idempotent: every assistant turn except the
# last is "```python\n" + body.rstrip("\n") + "\n\n```". Final targets end with one newline, as the Kimi targets do (generated code
# changes only in its trailing newline, so the TPU verdict still applies).
fixed_hist = 0
FENCE = re.compile(r"\A```python\n(.*?)\n*```\Z", re.S)
for r in rows:
    m = r["messages"]
    for k, t in enumerate(m):
        if t["role"] != "assistant":
            continue
        body = FENCE.match(t["content"]).group(1)
        want = "```python\n" + body.rstrip("\n") + ("\n```" if k == len(m) - 1 else "\n\n```")
        if want != t["content"]:
            fixed_hist += k != len(m) - 1
            t["content"] = want
ids = collections.Counter(r["example_id"] for r in rows)
assert max(ids.values()) == 1, [k for k, n in ids.items() if n > 1][:3]
(D / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
(D / "dev_items.jsonl").write_text("".join(json.dumps({"example_id": r["example_id"], "task": r["task"], "family": r["family"], "chain": chain(r)}) + "\n" for r in dev))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
man = {"rows": len(rows), "parts": {"v6_a6_minus_4": len(base), "kimi_kn0_pass": len(scaled), "generated_train": len(gen_train), "a6_rule2_dropped": len(a6), "history_turns_rewrapped": fixed_hist},
       "dev_holdout": {"rows": len(dev), "chains": len(dev_chains)}, "by_kind": dict(collections.Counter(r["kind"] for r in rows)),
       "by_teacher": dict(collections.Counter(r["teacher"] for r in rows)),
       "below_0.7x_rows": sum((r.get("speed_vs_xla") or 1) < 0.7 for r in rows),
       "max_target_chars": max(len(r["messages"][-1]["content"]) for r in rows),
       "sources": {"v6_a6": "hruday/kimi-k31@c8e64e3 kimi/v6/v6_train.jsonl sha256 d9271a92...", "kn0": "oracle-results/kimi-kn0-s1..4-p1",
                   "generated": "aarav/dataset-v4@de8f43a dataset/v7_gen/v7_gen_rows.jsonl sha256 ff739ee6..."},
       "files": {"train.jsonl": sha(D / "train.jsonl"), "dev_items.jsonl": sha(D / "dev_items.jsonl")}}
(D / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps(man, indent=1))
