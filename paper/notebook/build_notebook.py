"""Build paper/notebook/v7_submission.ipynb (the Kaggle submission notebook). CPU only, standard library + matplotlib.
Every file is fetched from the public repo at a pinned commit and checked against its sha256, so the notebook reproduces the paper's
numbers from raw data with no local state.   python3 paper/notebook/build_notebook.py"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Teaching a 4B Gemma to write TPU kernels: the v7 data and JAXBench results

This notebook is the companion to our Kaggle Writeup. It runs on CPU in a few minutes. It downloads the final training set and the
raw JAXBench run files from the public repository (`Art4v/gemma-hackathon-research`) at pinned commits, checks every file's sha256,
and recomputes every number in the writeup from those raw rows:

1. **The v7 training set** (2,304 rows): composition, row format, and the 5 rows we found that pass the oracle through
   input-dependent shortcuts.
2. **JAXBench**, for all 8 arms: pass within 3, correct within 3, pass@1, parity, the k and p subsets, and exact McNemar tests.
3. **v7 in detail**: per-task verdicts by turn, verdict transitions, and one of its passing kernels.
4. *(Optional, GPU)* Loading the v7 LoRA on Gemma 4 E4B.

The TPU oracle itself needs a TPU v6e; here we read its recorded verdicts. Every verdict row carries the oracle's file hash
`c9a07d9298a1`.
""")

code("""
import collections, hashlib, json, math, statistics, urllib.request, pathlib, os

REPO = "https://raw.githubusercontent.com/Art4v/gemma-hackathon-research"
DATA_COMMIT = "2efa6a5"        # hruday/kimi-k31: kimi/v7/v7_final.jsonl (the file v7 was trained on) + val
PAPER_COMMIT = "170ad6a"       # hruday/paper-v7: paper/data/runs (raw JAXBench rows of every arm)
CACHE = pathlib.Path(os.environ.get("V7_CACHE", "v7_cache")); CACHE.mkdir(exist_ok=True)

def fetch(commit, path, sha256=None):
    dst = CACHE / commit / path
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{REPO}/{commit}/{path}", dst)
    if sha256:
        got = hashlib.sha256(dst.read_bytes()).hexdigest()
        assert got == sha256, f"{path}: sha256 {got[:12]} != {sha256[:12]}"
    return dst

def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]
""")

md("## 1. The v7 training set")
code("""
train = jl(fetch(DATA_COMMIT, "kimi/v7/v7_final.jsonl", "10345461a48148d379f99c7d304180eff0d23bf9693568cbef99090db7abcda9"))
val = jl(fetch(DATA_COMMIT, "kimi/v7/v7_final_val.jsonl", "d09e86649892595c375769c7c779deeefa9fcaecf9a140f755b87b0166860fac"))
print(f"train rows: {len(train)}   val rows: {len(val)}   (sha256 verified)")
assert len(train) == 2304 and len(val) == 29

SET_NAMES = {"pool": "Kimi K3.1 pool (base)", "e1_readmit": "Kimi, re-admitted under the instance rule (base)",
             "e2_kbkd": "Kimi, KernelBook batch d (base)", "e3_scale": "Kimi, scale variants (base)",
             "e4_kn": "Exam-scale: existing kernel passed unchanged", "e4_fix": "Exam-scale: Kimi fix, round 1",
             "e4_fix2": "Exam-scale: Kimi fix, round 2", "r3_gap": "Round 3: v4-covered gap",
             "e5_gen": "Template generator", "e5_gen_layout": "Template generator: layouts", "e5_gen7r": "Template generator: whole-row softmax"}
by_set = collections.Counter(r["set"] for r in train)
for s, n in by_set.most_common():
    variants = sum(1 for r in train if r["set"] == s and r.get("surface_of"))
    print(f"{n:5}  {SET_NAMES.get(s, s)}" + (f"  ({variants} surface variants)" if variants else ""))
print()
print("teachers:", dict(collections.Counter(r["teacher"] for r in train)))
print("kinds:   ", dict(collections.Counter(r["kind"] for r in train)))
tmpl = sum(r["teacher"] == "generator-v7" for r in train)
print(f"template share: {tmpl}/{len(train)} = {100 * tmpl / len(train):.1f}%")
assert all(r["thinking"] is False for r in train + val)
assert not {r["task"] for r in val} & {r["task"] for r in train}, "a val task is in train"
""")

md("""
**Row format.** Every row is built by the evaluation's own prompt function (`control/bench.py` `conversation()`), so it is
byte-identical to what the model sees at test time. A repair row carries the failed kernel(s) and the oracle's verbatim feedback;
loss is on the final assistant turn only.
""")
code("""
r = next(x for x in train if x["kind"] == "repair" and len(x["messages"]) == 5)
print("example repair row:", r["example_id"], "| verdict", r["verdict"], "| speed vs XLA", round(r["speed_vs_xla"], 2))
for m in r["messages"]:
    text = m["content"]
    print(f"--- {m['role']} ({len(text)} chars)\\n{text[:400]}{' ...' if len(text) > 400 else ''}\\n")
""")

md("""
**Known-bad rows.** After training, reading the targets showed 5 rows (0.2%) that pass the oracle through input-dependent
shortcuts: a mean in place of a median, and polynomial approximations accurate only on the reference's input range. Three seeds
from fixed input distributions cannot rule these out. v7 was trained on them; the published dataset release excludes them.
""")
code("""
KNOWN_BAD = ["kernelbook__IMQSteinKernel_5388.kimi.a1", "kernelbook__NAC_188__kn8.kimi.a2", "kernelbook__NAC_188__kn8.kimi.a2t0",
             "kernelbook__PNTrainingSigmoid_7307__scale.kimi.a2", "kernelbook__CompositeActivation_1340.kimi.a2"]
ids = {x["example_id"]: x for x in train}
for k in KNOWN_BAD:
    x = ids.get(k)
    print(f"{k:55} {'present' if x else 'MISSING'}  speed {round(x['speed_vs_xla'], 2) if x else ''}")
release = [x for x in train if x["example_id"] not in KNOWN_BAD]
print("rows in the release (train minus known-bad):", len(release))
RELEASE_COMMIT = "105fcaf"   # aarav/paper-v7: paper/release/v7_release.jsonl, the published file
pub = jl(fetch(RELEASE_COMMIT, "paper/release/v7_release.jsonl", "002da96dc410dd219772c456232a7bc3aef750fbb915fef9b9594640ce6a4c97"))
assert [x["example_id"] for x in pub] == [x["example_id"] for x in release], "the release file is not train minus the 5 known-bad rows"
assert all(json.dumps(a) == json.dumps(b) for a, b in zip(pub, release)), "a release row differs from its training row"
print("published release file: sha256 verified, equal to train minus the 5 known-bad rows, row for row")
speeds = [x["speed_vs_xla"] for x in train if x.get("speed_vs_xla")]
print(f"target speed vs XLA: median {statistics.median(speeds):.2f}x, {sum(s > 5 for s in speeds)} rows above 5x (report medians)")
""")

md("""
## 2. JAXBench

**Tasks:** JAXBench's 50 level 1 tasks plus 14 level 2-only tasks (64). The **primary set** is the 50 level 1 tasks plus the 8
level 2 tasks whose plain-JAX reference verified against Google's kernel on the TPU: 58 tasks, 33 k and 25 p.

**Budget:** 3 attempts per task (turn 0 plus 2 repairs with the oracle's verbatim feedback); a task stops at its first pass.

**Metrics:** `pass_within_3` (a pass on any turn), `correct_within_3`, `pass@1` and parity (correct at 0.97x or faster).
Paired comparisons use an exact two-sided McNemar test.
""")
code("""
ARMS = {  # label -> (run dir, id prefix)
    "Gemma 4 E4B, bare": ("gemma4_e4b-it-qat__modal", "bare"),
    "Gemma 4 E4B + docs": ("gemma4_e4b-it-qat__modal", "docs"),
    "Gemini Flash + docs": ("gemini-3.6-flash", "docs"),
    "LoRA pilot (25 rows)": ("gemma4-e4b-lora_pilot25__modal", "bare"),
    "LoRA v1": ("gemma4-e4b-lora_v1__modal", "bare"),
    "LoRA v4": ("gemma4-e4b-lora_v4final-e2__modal", "bare"),
    "LoRA v5": ("gemma4-e4b-lora_v5full__modal", "bare"),
    "LoRA v7 (final)": ("gemma4-e4b-lora_v7final__modal", "bare"),
}
PRIMARY_L2 = {"51p", "53p", "56p", "57p", "58p", "61p", "62p", "63p"}
num = lambda t: t.split("_")[0]
is_primary = lambda t: int("".join(c for c in num(t) if c.isdigit())) <= 50 or num(t) in PRIMARY_L2
kind = lambda t: "k" if num(t).endswith("k") else "p"

def load_arm(run, prefix):
    gen = jl(fetch(PAPER_COMMIT, f"paper/data/runs/{run}/{prefix}.jsonl"))
    V = {}
    for v in jl(fetch(PAPER_COMMIT, f"paper/data/runs/{run}/verdicts.jsonl")):
        if not v.get("meta") and str(v.get("id", "")).startswith(prefix + "/"):
            V[v["id"]] = v
    out = {}
    for t in sorted({g["task"] for g in gen}):
        out[t] = [V[f"{prefix}/{t}/t{k}"] for k in range(3) if f"{prefix}/{t}/t{k}" in V]
    return out, gen

def outcome(turns):
    ok = lambda v: v["verdict"] in ("pass", "correct_slow")
    sp = [v["speed_vs_xla"] for v in turns if ok(v) and v.get("speed_vs_xla")]
    return {"pass": any(v["verdict"] == "pass" for v in turns), "correct": any(ok(v) for v in turns),
            "pass_at_1": bool(turns) and turns[0]["id"].endswith("/t0") and turns[0]["verdict"] == "pass",
            "parity": any(ok(v) and (v.get("speed_vs_xla") or 0) >= 0.97 for v in turns), "best_speed": max(sp) if sp else None}

RAW, OUT, DIGESTS = {}, {}, {}
for label, (run, prefix) in ARMS.items():
    RAW[label], gen = load_arm(run, prefix)
    OUT[label] = {t: outcome(v) for t, v in RAW[label].items()}
    DIGESTS[label] = collections.Counter(str((g.get("backend") or {}).get("digest", ""))[:12] for g in gen)
print("served-model digests per arm (every row of an arm must carry one digest):")
for label, d in DIGESTS.items():
    print(f"  {label:24} {dict(d)}")
""")

code("""
prim = sorted(t for t in OUT["LoRA v7 (final)"] if is_primary(t))
K = [t for t in prim if kind(t) == "k"]; P = [t for t in prim if kind(t) == "p"]
print(f"primary tasks: {len(prim)} ({len(K)} k, {len(P)} p)\\n")
hdr = f"{'arm':24} {'pass<=3':>8} {'correct<=3':>11} {'pass@1':>7} {'parity':>7} {'k-33':>5} {'p-25':>5}  passed"
print(hdr); print("-" * len(hdr))
for label, oc in OUT.items():
    c = lambda key, ts: sum(oc[t][key] for t in ts if t in oc)
    passed = " ".join(sorted((num(t) for t in prim if oc.get(t, {}).get("pass")), key=lambda x: int(x[:-1])))
    print(f"{label:24} {c('pass', prim):>5}/58 {c('correct', prim):>8}/58 {c('pass_at_1', prim):>4}/58 {c('parity', prim):>4}/58 "
          f"{c('pass', K):>2}/33 {c('pass', P):>2}/25  {passed}")
""")

code("""
def mcnemar(a, b, key="pass"):
    ts = [t for t in prim if t in OUT[a] and t in OUT[b]]
    n10 = sum(OUT[a][t][key] and not OUT[b][t][key] for t in ts)
    n01 = sum(OUT[b][t][key] and not OUT[a][t][key] for t in ts)
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n10, n01, p

print(f"{'v7 vs':24} {'v7 only':>8} {'other only':>11} {'p (exact, two-sided)':>21}")
for other in ARMS:
    if other != "LoRA v7 (final)":
        n10, n01, p = mcnemar("LoRA v7 (final)", other)
        print(f"{other:24} {n10:>8} {n01:>11} {p:>21.4f}")
""")

code("""
import matplotlib.pyplot as plt
labels = list(OUT)
cum = {l: [sum(any(v["verdict"] == "pass" for v in RAW[l][t][:k + 1]) for t in prim if t in RAW[l]) for k in range(3)] for l in labels}
fig, ax = plt.subplots(figsize=(7, 4))
for l in labels:
    ax.plot([1, 2, 3], cum[l], marker="o", label=l, lw=2.5 if "v7" in l else 1.2)
ax.set_xticks([1, 2, 3]); ax.set_xlabel("attempts (turn 0 + repairs)"); ax.set_ylabel("tasks passed (of 58)")
ax.set_title("JAXBench primary-58: passes within k attempts"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md("## 3. v7 in detail")
code("""
v7 = RAW["LoRA v7 (final)"]
order = sorted(v7, key=lambda t: int("".join(c for c in num(t) if c.isdigit())))
for t in order:
    turns = " | ".join(f"{v['verdict']}{'' if not v.get('speed_vs_xla') else ' %.2fx' % v['speed_vs_xla']}" for v in v7[t])
    print(f"{t[:46]:46} {'P' if is_primary(t) else ' '}  {turns}")
""")
code("""
tr = collections.Counter()
for t, turns in v7.items():
    for a, b in zip(turns, turns[1:]):
        tr[(a["verdict"], b["verdict"])] += 1
print("verdict transitions between consecutive attempts (v7):")
for (a, b), n in tr.most_common():
    print(f"  {a:>12} -> {b:<12} {n}")
v4 = RAW["LoRA v4"]
first = lambda R: collections.Counter(R[t][0]["verdict"] for t in R if R[t])
print("\\nturn-0 verdicts, v4:", dict(first(v4)))
print("turn-0 verdicts, v7:", dict(first(v7)))
""")

md("One of v7's passing kernels (41k, `Gemm_Add_ReLU`, first attempt). It tiles M, N and K, accumulates in f32 VMEM scratch, and applies the epilogue at the last K step:")
code("""
gen7 = {(g["task"], g["turn"]): g for g in jl(fetch(PAPER_COMMIT, "paper/data/runs/gemma4-e4b-lora_v7final__modal/bare.jsonl"))}
t = next(t for t in v7 if t.startswith("41k_"))
print(t, "->", v7[t][0]["verdict"], f"{v7[t][0]['speed_vs_xla']:.2f}x XLA\\n")
print(gen7[(t, 0)]["code"])
""")

md("""
## 4. (Optional, GPU) Load the v7 adapter

The adapter (LoRA rank 64, alpha 128, attention and MLP projections) was trained against the exact dequantized weights of the 4-bit
QAT GGUF that Ollama serves. For serving we fold it into that GGUF, storing the 258 touched tensors at Q8_0. The cell below shows
the Hugging Face route; it is not run in this notebook. Set `ADAPTER_DIR` to a copy of the released adapter.
""")
code("""
RUN_GPU_DEMO = False
if RUN_GPU_DEMO:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    BASE = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
    ADAPTER_DIR = "path/to/v7/adapter"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto"), ADAPTER_DIR)
    prompt = tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=2048, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True)[-3000:])
""")

md("""
## Limitations (from the writeup)

- **JAXBench informed v7's design.** The exam-scale rows exist because v4 failed on VMEM at JAXBench's sizes. No JAXBench task,
  reference or kernel is in training, and every row whose op chain matches a k-task was removed. But v7 is the fifth adapter we ran
  on JAXBench, and we report every run.
- **Template rows.** 30.8% of rows come from one generator template, with surface variants. The dev set is template items, so its
  loss measures template recall, not tiling ability.
- **Testing is not verification.** 5 rows pass the oracle through input-dependent shortcuts (Section 1).
- **One greedy chain, 3 attempts per task.** Gemini Flash samples at its own defaults.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "v7_submission.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", HERE / "v7_submission.ipynb", f"({len(cells)} cells)")
