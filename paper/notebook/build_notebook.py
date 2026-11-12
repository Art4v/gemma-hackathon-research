"""Build paper/notebook/submission.ipynb (the Kaggle submission notebook). CPU only, standard library + matplotlib.
Every file is fetched from the public repo at a pinned commit and checked against its sha256, so the notebook reproduces the paper's
numbers from raw data with no local state.   python3 paper/notebook/build_notebook.py"""
import json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md("""
# Teaching a 4B Gemma to write TPU kernels: the training data and JAXBench results

This notebook is the companion to our Kaggle Writeup. It runs on CPU in a few minutes. It downloads the training set (GitHub release `dataset-final`)
and the raw JAXBench run files (the public repository `Art4v/gemma-hackathon-research`, at a pinned commit), checks every file's sha256,
and recomputes every number in the writeup from those raw rows:

1. **The training set** (2,304 rows): composition, row format, and the 5 rows we found that pass the oracle through
   input-dependent shortcuts.
2. **JAXBench**, for all 4 arms: pass within 3, correct within 3, pass@1, parity, the k and p subsets, and exact McNemar tests.
3. **Our adapter in detail**: per-task verdicts by turn, verdict transitions, and one of its passing kernels.
4. *(Optional, GPU)* Loading our LoRA on Gemma 4 E4B.

The TPU oracle itself needs a TPU v6e; here we read its recorded verdicts. Every verdict row carries the oracle's file hash
`c9a07d9298a1`.
""")

md("""
## Setup

This cell defines the download helpers. Nothing is read from local state.

- `fetch_release` downloads a file from the GitHub release `dataset-final` and stops if its sha256 differs from the value written here.
- `fetch` downloads a file from the repository at one fixed commit, so the run rows cannot change under the notebook.
- Files are kept in a cache folder (`nb_cache`, or `NB_CACHE` if set), so a second run downloads nothing.

The notebook needs internet access for the first run (on Kaggle: Settings, Internet on). The downloads total about 27 MB.
""")

code("""
import collections, hashlib, json, math, statistics, urllib.request, pathlib, os

REPO = "https://raw.githubusercontent.com/Art4v/gemma-hackathon-research"
RELEASE = "https://github.com/Art4v/gemma-hackathon-research/releases/download/dataset-final"
PAPER_COMMIT = "52b8b6e"       # paper/data/runs: raw JAXBench rows of every arm
CACHE = pathlib.Path(os.environ.get("NB_CACHE", "nb_cache")); CACHE.mkdir(exist_ok=True)

def fetch(commit, path, sha256=None):
    dst = CACHE / commit / path
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{REPO}/{commit}/{path}", dst)
    if sha256:
        got = hashlib.sha256(dst.read_bytes()).hexdigest()
        assert got == sha256, f"{path}: sha256 {got[:12]} != {sha256[:12]}"
    return dst

def fetch_release(name, sha256):
    dst = CACHE / "release" / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{RELEASE}/{name}", dst)
    got = hashlib.sha256(dst.read_bytes()).hexdigest()
    assert got == sha256, f"{name}: sha256 {got[:12]} != {sha256[:12]}"
    return dst

def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]
""")

md("""
## 1. The training set

The adapter was trained on `train_as_trained.jsonl`. The development set `val.jsonl` was held out. Each row is one conversation:
a system prompt, a plain-JAX reference as the user turn, and a Pallas kernel as the final assistant turn. The target of every row
was judged correct by the TPU oracle, or is a renamed copy of a kernel that was.
""")

md("""
The cell below loads the two files and counts the rows by source group, by who wrote the target, and by kind.

How to read the output:

- **Source groups.** "base" groups are teacher kernels on the original problems. "Exam-scale" groups are matmul problems enlarged to
  production sizes. "Template generator" groups are kernels emitted from our tiling template; a surface variant is the same kernel
  with renamed arguments, configuration keys and docstring.
- **Teachers.** `kimi-k3.1` is the open-weights teacher model, and `generator-v7` is the template generator.
- **Kinds.** `single` is a one-turn row whose target passed. `correct_slow` is a one-turn row whose target is correct but below
  0.7x XLA. `repair` is a chain of one or two failed attempts ending in a pass, and `repair_slow` is a chain ending in a correct
  but slow kernel. That gives 1,720 single-turn rows and 584 repair chains.

The last two lines check that every row was built with thinking off, and that no development task also appears in training.
""")

code("""
train = jl(fetch_release("train_as_trained.jsonl", "10345461a48148d379f99c7d304180eff0d23bf9693568cbef99090db7abcda9"))  # exactly what the adapter trained on
val = jl(fetch_release("val.jsonl", "d09e86649892595c375769c7c779deeefa9fcaecf9a140f755b87b0166860fac"))
print(f"train rows: {len(train)}   val rows: {len(val)}   (sha256 verified)")
assert len(train) == 2304 and len(val) == 29

SET_NAMES = {"pool": "Kimi K3.1 pool (base)", "e1_readmit": "Kimi, re-admitted under the instance rule (base)",
             "e2_kbkd": "Kimi, KernelBook batch d (base)", "e3_scale": "Kimi, scale variants (base)",
             "e4_kn": "Exam-scale: existing kernel passed unchanged", "e4_fix": "Exam-scale: Kimi fix, round 1",
             "e4_fix2": "Exam-scale: Kimi fix, round 2", "r3_gap": "Third-round teacher kernels",
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

The cell below prints one repair row with two attempts: the system prompt, the reference, the failed kernel, the oracle's feedback
and the corrected kernel. Each message is cut to its first 400 characters.
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
from fixed input distributions cannot rule these out. The adapter was trained on them; the published dataset release excludes them.

The cell below confirms that the 5 rows are in the file the adapter trained on, prints the speed each one was credited with, and
checks that the published `train.jsonl` equals the as-trained file with exactly those 5 rows removed. It also prints the median
target speed; a few rows have very large speed ratios, so the median is the number to quote.
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
pub = jl(fetch_release("train.jsonl", "002da96dc410dd219772c456232a7bc3aef750fbb915fef9b9594640ce6a4c97"))  # the published training set
assert [x["example_id"] for x in pub] == [x["example_id"] for x in release], "the release file is not train minus the 5 known-bad rows"
assert all(json.dumps(a) == json.dumps(b) for a, b in zip(pub, release)), "a release row differs from its training row"
print("published train.jsonl: sha256 verified, equal to the as-trained file minus the 5 known-bad rows, row for row")
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

md("""
### Loading the run rows

Each arm has two files, with one row per attempt in each:

- a **generation file** with the prompt hash, the decoding settings, the model's full response and the extracted code;
- a **verdict file** with the oracle's verdict for that attempt, its feedback line and, for correct kernels, the speed against XLA.

An arm is one model with one prompt. A task has up to three attempts, named `t0`, `t1` and `t2`. The cell joins the two files by
attempt id and reduces each task to four yes/no outcomes: passed, correct at any speed, passed on the first attempt, and correct
at parity (0.97x XLA or faster).

The printed digests identify the model that was actually served. The two base arms share one digest (the official 4-bit Gemma 4
E4B build), and our arm has a different one (the same build with our adapter folded in). Gemini Flash is an API model and has no
digest, so its entry is empty.
""")

code("""
OURS = "Gemma 4 E4B + our LoRA"
ARMS = {  # label -> (run dir, id prefix)
    "Gemma 4 E4B, bare": ("gemma4_e4b-it-qat__modal", "bare"),
    "Gemma 4 E4B + docs": ("gemma4_e4b-it-qat__modal", "docs"),
    "Gemini Flash + docs": ("gemini-3.6-flash", "docs"),
    "Gemma 4 E4B + our LoRA": ("gemma4-e4b-lora__modal", "bare"),
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

md("""
### Main table

One line per arm, on the 58 primary tasks. `pass<=3` is the headline metric: tasks with a pass on any of the three attempts.
`correct<=3` also counts kernels that are correct but slower than 0.7x XLA. `k-33` and `p-25` split the passes into the 33
operator-chain tasks ported from KernelBench and the 25 production-kernel tasks. The last column lists the tasks each arm passed.
""")

code("""
prim = sorted(t for t in OUT[OURS] if is_primary(t))
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

md("""
### Paired tests

Every arm ran the same 58 tasks, so two arms are compared task by task. Only the tasks where they disagree carry information:
"ours only" counts tasks our adapter passed and the other arm did not, and "other only" counts the reverse. The exact McNemar
test asks how likely a split at least this uneven would be if both arms were equally good. A small p means the difference is
unlikely to be chance.
""")

code("""
def mcnemar(a, b, key="pass"):
    ts = [t for t in prim if t in OUT[a] and t in OUT[b]]
    n10 = sum(OUT[a][t][key] and not OUT[b][t][key] for t in ts)
    n01 = sum(OUT[b][t][key] and not OUT[a][t][key] for t in ts)
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n10, n01, p

print(f"{'ours vs':24} {'ours only':>9} {'other only':>11} {'p (exact, two-sided)':>21}")
for other in ARMS:
    if other != OURS:
        n10, n01, p = mcnemar(OURS, other)
        print(f"{other:24} {n10:>9} {n01:>11} {p:>21.4f}")
""")

md("""
### Passes by attempt

The same data as a curve: how many of the 58 primary tasks each arm has passed after one, two and three attempts. The two base
Gemma lines lie on top of each other at zero.
""")

code("""
import matplotlib.pyplot as plt
labels = list(OUT)
cum = {l: [sum(any(v["verdict"] == "pass" for v in RAW[l][t][:k + 1]) for t in prim if t in RAW[l]) for k in range(3)] for l in labels}
fig, ax = plt.subplots(figsize=(7, 4))
for l in labels:
    ax.plot([1, 2, 3], cum[l], marker="o", label=l, lw=2.5 if l == OURS else 1.2)
ax.set_xticks([1, 2, 3]); ax.set_xlabel("attempts (turn 0 + repairs)"); ax.set_ylabel("tasks passed (of 58)")
ax.set_title("JAXBench primary-58: passes within k attempts"); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md("""
## 3. Our adapter in detail
""")

md("""
The cell below prints every attempt our adapter made, one task per line. `P` marks the 58 primary tasks. Attempts are separated
by `|`, and a line stops at the first pass. A number such as `1.37x` is the kernel's speed against XLA, which is only measured
when the kernel ran.

The verdicts, in the order the oracle checks them:

- `not_a_kernel`: the main computation is not inside a `pallas_call`;
- `compile`: the code fails while it is traced or compiled;
- `vmem`: the blocks do not fit the TPU's on-chip memory;
- `mosaic`: the TPU backend rejects the kernel;
- `wrong`: it runs, but the output does not match the reference;
- `correct_slow`: correct, but slower than 0.7x XLA;
- `pass`: correct and at least 0.7x XLA.
""")

code("""
ours = RAW[OURS]
order = sorted(ours, key=lambda t: int("".join(c for c in num(t) if c.isdigit())))
for t in order:
    turns = " | ".join(f"{v['verdict']}{'' if not v.get('speed_vs_xla') else ' %.2fx' % v['speed_vs_xla']}" for v in ours[t])
    print(f"{t[:46]:46} {'P' if is_primary(t) else ' '}  {turns}")
""")

md("""
### What repair changes

After a failed attempt the model sees its own kernel and the oracle's feedback line, then tries again. The cell below counts how
the verdict changes from one attempt to the next, over all 64 tasks. Rows where the verdict stays the same are repairs that did
not help. It also prints the verdicts of the first attempts alone.
""")

code("""
tr = collections.Counter()
for t, turns in ours.items():
    for a, b in zip(turns, turns[1:]):
        tr[(a["verdict"], b["verdict"])] += 1
print("verdict transitions between consecutive attempts (ours):")
for (a, b), n in tr.most_common():
    print(f"  {a:>12} -> {b:<12} {n}")
first = lambda R: collections.Counter(R[t][0]["verdict"] for t in R if R[t])
print("\\nturn-0 verdicts:", dict(first(ours)))
""")

md("""
One of our adapter's passing kernels (41k, `Gemm_Add_ReLU`, first attempt). It tiles M, N and K, accumulates in f32 VMEM scratch, and applies the epilogue at the last K step:

Imports come first, then a helper `_tile` that picks the largest aligned tile dividing an axis, then the kernel body, then
`workload`, which sets up the grid and block shapes. The limit of 124 MiB on the last lines is the VMEM budget the training rows use.
""")

code("""
gen_rows = {(g["task"], g["turn"]): g for g in jl(fetch(PAPER_COMMIT, "paper/data/runs/gemma4-e4b-lora__modal/bare.jsonl"))}
t = next(t for t in ours if t.startswith("41k_"))
print(t, "->", ours[t][0]["verdict"], f"{ours[t][0]['speed_vs_xla']:.2f}x XLA\\n")
print(gen_rows[(t, 0)]["code"])
""")

md("""
## 4. (Optional, GPU) Load our adapter

The adapter (LoRA rank 64, alpha 128, attention and MLP projections) was trained against the exact dequantized weights of the 4-bit
QAT GGUF that Ollama serves. For serving we fold it into that GGUF, storing the 258 touched tensors at Q8_0. The cell below shows
the Hugging Face route; it is not run in this notebook. Set `ADAPTER_DIR` to a copy of the released adapter.

The adapter files are in the GitHub release `model-final` of `Art4v/gemma-hackathon-research` (`adapter_model.safetensors`,
`adapter_config.json` and a model card with checksums). Loading it on the Hugging Face checkpoint gives weights that are close to,
but not identical to, the model we scored, which uses the exact 4-bit values. The flag `RUN_GPU_DEMO` is off, so the cell does
nothing unless you turn it on.
""")

code("""
RUN_GPU_DEMO = False
if RUN_GPU_DEMO:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    BASE = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
    ADAPTER_DIR = "path/to/adapter"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto"), ADAPTER_DIR)
    prompt = tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=2048, do_sample=False)
    print(tok.decode(out[0], skip_special_tokens=True)[-3000:])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(HERE / "submission.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", HERE / "submission.ipynb", f"({len(cells)} cells)")
