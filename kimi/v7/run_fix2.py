"""[ROUND 2] v7 Kimi fixes, second round (fix2): repair turns over the WHOLE history (original kernel + kn0 feedback, then the round-1 fix +
its feedback) for round-1 fixes that failed step 1 / TPU or were slow, plus speed fixes for scaled kernels that were correct but slow.
v7 Kimi fixes: for every scaled item whose existing verified kernel failed at exam scale on the TPU (kimi-kn0 verdict vmem / wrong / compile /
mosaic / runtime / timeout), ask Kimi K3.1 for a fix as a REPAIR TURN: the constructor's kernel_user(ref, [(failed kernel, oracle feedback)]), the
same prompt our repair rounds used, with the TPU-measured tiling policy appended to the system prompt (Karthik's sweeps on a v6e-1, Nov 11).
  PROFILE=global|us SHARD=i NSHARDS=n CAP_USD=... [ONLY=ids.txt] python run_fix.py        (resumable; out/fix1/)"""
import glob, json, os, pathlib, sys, zlib
V7 = pathlib.Path(__file__).resolve().parent
K = pathlib.Path.home() / "Projects/gemma-work/wt-kimi/kimi/scripts"
sys.path.insert(0, str(K))
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from bedrock_kimi import BedrockKimi, Request  # noqa: E402
from constructor.prompts import kernel_user  # noqa: E402

PROFILE, SHARD, NSH = os.environ["PROFILE"], int(os.environ["SHARD"]), int(os.environ["NSHARDS"])
CAP, WORKERS = float(os.environ["CAP_USD"]), int(os.environ.get("WORKERS", "12"))
PRICES = {"global": (4.0, 20.0), "us": (4.4, 22.0)}[PROFILE]
REPAIR = {"compile", "not_a_kernel", "mosaic", "vmem", "runtime", "wrong", "timeout"}
POLICY = """

TILING POLICY (required here: the inputs are at full JAXBench scale, so the operands do not fit in VMEM whole; these settings were measured on a TPU v6e):
- In every pallas_call set the TPU compiler params' vmem_limit_bytes to 124 * 1024 * 1024.
- Matmul and batched matmul: tile M, N and K with (tm, tn, tk) = (2048, 2048, 1024), using a smaller tile only where a dimension is smaller. Grid (M // tm, N // tn, K // tk) with the K axis last and marked arbitrary; accumulate in an f32 VMEM scratch buffer, zero it when k == 0, and apply the epilogue and store the output only at the last k.
- If the epilogue reduces over the matmul's output row (sum, max, mean, softmax, logsumexp or a norm over N), the block must span ALL of N (tn = N) so the reduction sees the whole row; then use tm = 1024 and tk = 512 (tk = 256 when N >= 8192), and never a tk below 128.
- Never fall back to a whole-dimension block when a size does not divide a tile; pick a tile that divides it.
"""
items = {}
for f in ("kn_items_existing.jsonl", "kn_items_existing_8.jsonl", "kn_items_existing_p.jsonl", "kn_items_v4only.jsonl"):
    for l in open(V7 / f):
        x = json.loads(l); items[x["id"]] = x
ver = json.load(open(V7 / "kn0_verdicts.json"))["verdicts"]
halt = lambda x: x.get("verdict") == "harness_oom" and any(h in (x.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))
k1 = {}
for g in glob.glob(str(pathlib.Path.home() / "Projects/gemma-work/tpu/ops1/repo/oracle/jobs/kimi-kn1-*/results/*.jsonl")):
    for l in open(g):
        if l.strip() and '"meta": true' not in l:
            x = json.loads(l)
            if not halt(x):
                k1[x["id"].split("/", 1)[1]] = x
f1 = {}
for g in glob.glob(str(V7 / "out/fix1/k31_fix1_*.jsonl")):
    for l in open(g):
        d = json.loads(l)
        if d.get("ok") and d.get("code"):
            f1[d["problem_id"]] = d
st1 = {json.loads(l)["id"].split("/", 1)[1]: json.loads(l) for l in open(V7 / "out/fix1/cpu_scored.jsonl")}
v4only = {json.loads(l)["id"] for l in open(V7 / "kn_items_v4only.jsonl")}
plan = {}   # id -> (category, history [(code, feedback)])
for iid, v in ver.items():
    if iid in v4only or iid not in items:
        continue
    pid = iid.split("/", 1)[1]; orig = items[iid]["code"]
    if v["verdict"] == "correct_slow":
        plan[iid] = ("slow_scaled", [(orig, v["feedback"])])
    elif v["verdict"] in REPAIR and pid in f1:
        h = [(orig, v["feedback"])]
        s1 = st1.get(pid)
        if s1 and s1["static"] != "static_ok":
            plan[iid] = ("fix1_step1_fail", h + [(f1[pid]["code"], s1.get("static_feedback") or s1["static"])])
        elif pid in k1 and k1[pid]["verdict"] in REPAIR:
            plan[iid] = ("fix1_tpu_fail", h + [(f1[pid]["code"], k1[pid]["feedback"])])
        elif pid in k1 and k1[pid]["verdict"] == "correct_slow":
            plan[iid] = ("fix1_slow", h + [(f1[pid]["code"], k1[pid]["feedback"])])
ONLY = set(open(os.environ["ONLY"]).read().split()) if os.environ.get("ONLY") else None
todo = [(i, p) for i, p in plan.items() if (ONLY is None or i in ONLY) and zlib.crc32(i.encode()) % NSH == SHARD]
outdir = V7 / "out/fix2"; outdir.mkdir(parents=True, exist_ok=True)
out = outdir / f"k31_fix2_{PROFILE}_s{SHARD}.jsonl"
done = set()
for f in outdir.glob("k31_fix2_*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if d.get("ok"):
            done.add(d["id"])
todo = [t for t in todo if t[0] not in done]
system = open(K / "system_prompts-1_with_docs.txt").read() + POLICY
print(f"v7 fixes ROUND 2 ({dict(__import__('collections').Counter(c for c, _ in plan.values()))}), {PROFILE} shard {SHARD}/{NSH}: {len(todo)} to generate ({len(done)} done), cap ${CAP}", flush=True)
k = BedrockKimi(model=f"{PROFILE}.moonshotai.kimi-k3.1", workers=WORKERS, cap_usd=CAP, max_output_tokens=48000, price_in=PRICES[0], price_out=PRICES[1],
                ledger=str(outdir / f"ledger_{PROFILE}_s{SHARD}.jsonl"), reasoning_fields={"reasoning_effort": "high"})
by = dict(todo)
f = out.open("a")


def save(resp):
    cat, hist = by[resp.key]; it = items[resp.key]; o = resp.output or {}
    f.write(json.dumps({"id": resp.key, "problem_id": resp.key.split("/", 1)[1], "task": it["task"], "attempt": len(hist) + 1, "category": cat,
                        "history": [{"code": c, "feedback": fb} for c, fb in hist], "ok": resp.ok, "model": resp.model, "profile": PROFILE,
                        "effort": "high", "prompt_version": "prompts-1+json_rule kernel_user(history) + v7 tiling policy",
                        "reasoning": o.get("reasoning", ""), "plan": o.get("plan"), "code": o.get("code"), "parsed_from": o.get("_parsed_from", "json"),
                        "text_if_unparsed": o.get("text") if not resp.ok else None, "usage": resp.usage, "error": resp.error}) + "\n")
    f.flush()


k.generate([Request(key=i, system=system, user=kernel_user(items[i]["ref_code"], p[1])) for i, p in todo],
           on_result=save)
print(f"FIX2 GEN DONE {PROFILE} s{SHARD}: spent ${k.ledger.spent:.2f}", flush=True)
