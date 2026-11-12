"""Independent final check of a v7 training file (and its val file). Read-only; prints findings.
usage: python check.py TRAIN VAL REPO_SCRIPTS_DIR BENCH_PY V3_DIR
"""
import ast, collections, difflib, importlib.util, json, pathlib, re, sys

TRAIN, VAL, SCRIPTS, BENCH, V3 = sys.argv[1:6]
sys.path.insert(0, SCRIPTS)
spec = importlib.util.spec_from_file_location("bench", BENCH); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
read = lambda p: [json.loads(l) for l in open(p) if l.strip()]
tr, va = read(TRAIN), read(VAL)
V3 = pathlib.Path(V3)
F = collections.defaultdict(list)          # finding -> examples
def flag(k, x):
    F[k].append(x)

FENCE = re.compile(r"\A```python\n(.*?)```\Z", re.S)
def ref_of(r):
    return re.findall(r"```python\n(.*?)\n```", r["messages"][1]["content"], re.S)[0]

# A. structure
for r in tr + va:
    m = r["messages"]
    roles = [x["role"] for x in m]
    if roles not in (["system", "user", "assistant"], ["system", "user", "assistant", "user", "assistant"], ["system", "user", "assistant", "user", "assistant", "user", "assistant"]):
        flag("A roles", (r["example_id"], roles))
    if r.get("thinking") is not False:
        flag("A thinking not False", r["example_id"])
    for x in m:
        if set(x) != {"role", "content"}:
            flag("A extra message keys", (r["example_id"], sorted(x)))
sysp = collections.Counter(r["messages"][0]["content"] for r in tr + va)
if len(sysp) != 1:
    flag("A system prompts differ", len(sysp))

# B. prompts rebuild through bench.conversation (turn 0) and the history turn
for r in tr + va:
    m = r["messages"]
    try:
        ref = ref_of(r)
    except IndexError:
        flag("B no ref block", r["example_id"]); continue
    c0 = b.conversation(ref, [], None)
    if c0[0]["content"] != m[0]["content"] or c0[1]["content"] != m[1]["content"]:
        flag("B turn-0 prompt != bench.conversation", r["example_id"])
    if len(m) > 3:
        # what the eval sends: the model's reply is "```python\n<code>\n```", extract_code keeps "<code>\n", conversation() wraps it again.
        # So from a history turn we recover the reply's code as body.rstrip("\n") + "\n" and rebuild the whole prefix through bench.
        hist = []
        for k in range(2, len(m) - 1, 2):
            body = FENCE.match(m[k]["content"]).group(1)
            code = body.rstrip("\n") + "\n"
            probe = b.conversation(ref, hist + [{"code": code, "feedback": "@@FB@@"}], None)
            pre, post = probe[k + 1]["content"].split("@@FB@@")
            fbt = m[k + 1]["content"]
            if not (fbt.startswith(pre) and fbt.endswith(post)):
                flag("B feedback turn wrapper != bench", (r["example_id"], k)); break
            hist.append({"code": code, "feedback": fbt[len(pre):len(fbt) - len(post)]})
        else:
            full = b.conversation(ref, hist, None)
            if [x["content"] for x in full] != [x["content"] for x in m[:-1]]:
                flag("B repair prefix != what the eval would send", r["example_id"])

# C. targets
for r in tr + va:
    t = r["messages"][-1]["content"]
    mm = FENCE.match(t)
    if not mm:
        flag("C target not one fenced block", r["example_id"]); continue
    body = mm.group(1)
    if "```" in body:
        flag("C nested fence", r["example_id"])
    if b.extract_code(t) != body:
        flag("C extract_code(target) != body", r["example_id"])
    try:
        tree = ast.parse(body)
    except SyntaxError as e:
        flag("C target does not parse", (r["example_id"], str(e)[:60])); continue
    if not any(isinstance(n, ast.FunctionDef) and n.name == "workload" for n in tree.body):
        flag("C no top-level workload()", r["example_id"])
    if "pallas_call" not in body:
        flag("C no pallas_call", r["example_id"])
    if len(t) > 18000:
        flag("C target > 18000 chars", (r["example_id"], len(t)))
    if not t.endswith("\n```") or t.endswith("\n\n```"):
        flag("C target not exactly one newline before fence", r["example_id"])

# D. duplicates and conflicting supervision
ids = collections.Counter(r["example_id"] for r in tr)
for k, n in ids.items():
    if n > 1:
        flag("D duplicate example_id", k)
by_prompt = collections.defaultdict(set)
for r in tr:
    by_prompt[json.dumps([x["content"] for x in r["messages"][:-1]])].add(r["messages"][-1]["content"])
for p, ts in by_prompt.items():
    if len(ts) > 1:
        flag("D same prompt, different targets", (json.loads(p)[1][:90].replace("\n", " "), len(ts)))
full = collections.Counter(json.dumps(r["messages"]) for r in tr)
for k, n in full.items():
    if n > 1:
        flag("D identical full rows", n)
by_ref = collections.defaultdict(list)
for r in tr:
    by_ref[ref_of(r)].append(r)
multi_turn0 = [(k, v) for k, v in by_ref.items() if sum(len(x["messages"]) == 3 for x in v) > 1]
for k, v in multi_turn0:
    flag("D same reference, >1 turn-0 rows", [x["example_id"] for x in v if len(x["messages"]) == 3][:3])

# E. repair rows: does the target actually change the failed attempt?
for r in tr:
    m = r["messages"]
    if len(m) < 5:
        continue
    a1, a2 = b.extract_code(m[-3]["content"]).strip(), b.extract_code(m[-1]["content"]).strip()
    if a1 == a2:
        flag("E repair target identical to failed attempt", r["example_id"])
    else:
        ratio = difflib.SequenceMatcher(None, a1, a2, autojunk=False).ratio() if len(a1) + len(a2) < 40000 else 0
        if ratio >= 0.99:
            flag("E repair target >=99% identical to failed attempt", (r["example_id"], round(ratio, 4)))

# F. leakage
val_tasks = {r["task"] for r in va}
for r in tr:
    if r["task"] in val_tasks:
        flag("F train task is a v7 val task", r["example_id"])
v1val = read(V3 / "dataset" / "final" / "val.jsonl")
v1fam = {e["family_id"] for e in v1val}
v1task = {e["task_name"] for e in v1val if "task_name" in e} | {r["task"] for r in read(V3 / "dataset" / "v4" / "final" / "val.jsonl")}
stem = lambda s: re.split(r"__(bf16|kn|scale|m1_|shape|gen7)", s)[0]
for r in tr:
    pid = r.get("problem_id", "")
    if r["task"] in v1task or stem(r["task"]) in {stem(t) for t in v1task}:
        flag("F train row from a dataset-v1 val task (by stem)", r["example_id"])
    if any(pid == f or pid.startswith(f + "/") or pid.startswith(f + "__") for f in v1fam):
        flag("F train row in a dataset-v1 val family", r["example_id"])
exam_name = re.compile(r"\b\d{1,2}[pk]_[A-Z][A-Za-z0-9_]+")
for r in tr + va:
    for x in r["messages"]:
        if x["role"] in ("user", "assistant") and exam_name.search(x["content"]):
            flag("F JAXBench-style task name in text", (r["example_id"], exam_name.search(x["content"]).group(0))); break
for r in tr:
    if "claude" in str(r.get("teacher", "")).lower():
        flag("F Claude-written target", r["example_id"])

# G. kinds and speed labels
for r in tr:
    s, k = r.get("speed_vs_xla"), r["kind"]
    if k in ("single", "repair") and (s is None or s < 0.7):
        flag("G pass-kind row below 0.7x or missing speed", (r["example_id"], k, s))
    if k in ("correct_slow", "repair_slow") and s is not None and s >= 0.7:
        flag("G slow-kind row at >= 0.7x", (r["example_id"], k, s))
    if r.get("verdict") not in ("pass", "correct_slow", None):
        flag("G verdict not pass/correct_slow", (r["example_id"], r.get("verdict")))

# H. agreed drops honoured
a6 = json.load(open(V3 / "dataset" / "v7_gen" / "a6_rule2_drops.json"))
dropped = set(a6["drop_generated_training_example_ids"])
dev = {json.loads(l)["example_id"] for l in open(V3 / "dataset" / "v7_fallback" / "dev_items.jsonl")}
for r in tr:
    if r["example_id"] in dropped:
        flag("H A6 rule-2 generated drop present", r["example_id"])
    if r["example_id"] in dev:
        flag("H dev item in training", r["example_id"])
    if r.get("problem_id") in {"kernelbook/Sentence_Maxpool_14501", "kernelbook/MetaBilinear_711", "kernelbook/CPC_15255", "kernelbook/QMaxPooling2d_9618"}:
        flag("H L3 drop present", r["example_id"])
    if any(k in r["example_id"] for k in ("GateLayer_1320", "Maxout_4302")) and r.get("set", "").startswith("e6"):
        flag("H kn0 A6 drop present", r["example_id"])

# I. exam near-duplicate (5-gram Jaccard on the team's normalised tokens; difflib skipped for speed)
from filter_neardup import exam_corpus, ngrams, normalise, alpha  # noqa: E402
corpus = [(t, ngrams(tok), ngrams(alpha(tok))) for t, _, tok in exam_corpus()]
worst = (0, None, None)
for ref, rows in by_ref.items():
    tok = normalise(ref); g, ga = ngrams(tok), ngrams(alpha(tok))
    for t, eg, ega in corpus:
        j = max(len(g & eg) / max(1, len(g | eg)), len(ga & ega) / max(1, len(ga | ega)))
        if j > worst[0]:
            worst = (j, rows[0]["example_id"], t)
        if j >= 0.5:
            flag("I reference near-dup of an exam file (Jaccard >= 0.5)", (rows[0]["example_id"], t, round(j, 3)))

# J. policy consistency (informational)
lim = collections.Counter()
for r in tr:
    t = r["messages"][-1]["content"]
    m = re.search(r"vmem_limit_bytes\s*=\s*([^,\)\n]+)", t)
    lim[(r.get("teacher"), (m.group(1).strip()[:24] if m else "none"))] += 1

print(f"train {len(tr)} rows, val {len(va)} rows")
print("kinds", dict(collections.Counter(r["kind"] for r in tr)), "| teachers", dict(collections.Counter(r.get("teacher") for r in tr)))
print("sets", dict(collections.Counter(r.get("set") for r in tr)))
print(f"distinct references {len(by_ref)}; max exam 5-gram Jaccard {worst[0]:.3f} ({worst[1]} -> {worst[2]})")
print("vmem_limit_bytes by teacher (top):", lim.most_common(8))
print("\nFINDINGS" if F else "\nNO FINDINGS")
for k in sorted(F):
    v = F[k]
    print(f"- {k}: {len(v)}  e.g. {v[:4]}")
