"""[NEW-problem pool: e1 re-admits, e2 kbkd, e3 scale variants; paths out/new/..., TPU jobs kimi-n<r>*]
Kimi K3.1 repair round R (2 or 3): for every problem whose latest attempt got a kernel-attributable oracle verdict (compile,
not_a_kernel, mosaic, vmem, runtime, wrong, timeout), send the constructor's own repair prompt, exactly as the Claude pipeline's rounds
2-3 did: one user message from kernel_user(ref_code, history) = the problem, then for each earlier attempt "[your previous answer]" +
the fenced kernel and "[oracle]" + "Oracle verdict: <feedback>" + the retry instruction. Same system prompt as round 1.
Excluded: pass, correct_slow (already correct), harness_oom / too_large_for_chip / harness_crash (not the kernel's fault), and attempts
still waiting for a TPU verdict.
  ROUND=2 PROFILE=global|us SHARD=i NSHARDS=n CAP_USD=... python run_repair.py   (resumable; out/rep<R>/)"""
import glob, json, os, pathlib, sys, zlib
P = pathlib.Path(__file__).resolve().parent
SC = P.parent
sys.path.insert(0, str(P))
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from bedrock_kimi import BedrockKimi, Request  # noqa: E402
from constructor.prompts import kernel_user  # noqa: E402  (the pipeline's repair rendering)

R = int(os.environ["ROUND"]); PROFILE, SHARD, NSH = os.environ["PROFILE"], int(os.environ["SHARD"]), int(os.environ["NSHARDS"])
CAP, WORKERS, EFFORT = float(os.environ["CAP_USD"]), int(os.environ.get("WORKERS", "12")), os.environ.get("EFFORT", "high")
PRICES = {"global": (4.0, 20.0), "us": (4.4, 22.0)}[PROFILE]
REPAIR = {"compile", "not_a_kernel", "mosaic", "vmem", "runtime", "wrong", "timeout"}


def jl(p):
    out = []
    for f in sorted(glob.glob(str(p))):
        for l in open(f):
            if l.strip():
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    return out

def _halt(d):  # Nov 11: harness_oom rows caused by an earlier kernel halting the TPU core are not verdicts (see tpu/crashloop.py)
    return d.get("verdict") == "harness_oom" and any(h in (d.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))


def attempts_for(r):
    """{problem_id: (code, verdict, feedback)} for attempt r: the generated code, then the CPU step-1 verdict or the TPU verdict."""
    gen_glob, cpu_file, tpu_glob = ((P / "out/new/k31_*.jsonl", P / "out/new/cpu_scored.jsonl", SC / "tpu/ops1/repo/oracle/jobs/kimi-n1*/results/*.jsonl")
                                    if r == 1 else (P / f"out/new/rep{r}/k31_rep{r}_*.jsonl", P / f"out/new/rep{r}/cpu_scored.jsonl",
                                                    SC / f"tpu/ops1/repo/oracle/jobs/kimi-n{r}*/results/*.jsonl"))
    code = {d["problem_id"]: d["code"] for d in jl(gen_glob) if d.get("ok") and d.get("code")}
    cpu = {d["problem_id"]: d for d in jl(cpu_file)}
    tpu = {d["id"].split("/", 1)[1]: d for d in jl(tpu_glob) if "id" in d and not d.get("meta") and not _halt(d)}
    out = {}
    for pid, c in code.items():
        s = cpu.get(pid)
        if s is None:
            continue
        if s["static"] != "static_ok":
            out[pid] = (c, s["static"], s.get("static_feedback") or "")
        elif pid in tpu:
            out[pid] = (c, tpu[pid].get("verdict"), tpu[pid].get("feedback") or "")
    return out


probs = {json.loads(l)["problem_id"]: json.loads(l) for f in ("new/new_problems.jsonl", "scale/scale_problems.jsonl") for l in open(P / f)}
ref = {json.loads(l)["problem_id"]: json.loads(l)["ref_code"] for f in ("new/new_refs.jsonl", "scale/scale_refs.jsonl") for l in open(P / f)}
hist = {r: attempts_for(r) for r in range(1, R)}
ONLY = set(open(os.environ["ONLY_FILE"]).read().split()) if os.environ.get("ONLY_FILE") else None
todo = []
for pid, p in probs.items():
    chain = [hist[r].get(pid) for r in range(1, R)]
    if any(a is None for a in chain) or chain[-1][1] not in REPAIR:
        continue
    if ONLY is not None and pid not in ONLY:
        continue
    if zlib.crc32(pid.encode()) % NSH != SHARD:
        continue
    todo.append((p, [(a[0], a[2]) for a in chain], [a[1] for a in chain]))
outdir = P / "out" / "new" / f"rep{R}"; outdir.mkdir(parents=True, exist_ok=True)
out = outdir / f"k31_rep{R}_{PROFILE}_s{SHARD}.jsonl"
done = {d["problem_id"] for d in jl(outdir / f"k31_rep{R}_*.jsonl") if d.get("ok")}
todo = [t for t in todo if t[0]["problem_id"] not in done]
system = open(P / "system_prompts-1_with_docs.txt").read()
print(f"repair round {R}, {PROFILE} shard {SHARD}/{NSH}: {len(todo)} to generate ({len(done)} done), cap ${CAP}", flush=True)
k = BedrockKimi(model=f"{PROFILE}.moonshotai.kimi-k3.1", workers=WORKERS, cap_usd=CAP, max_output_tokens=48000,
                price_in=PRICES[0], price_out=PRICES[1], ledger=str(outdir / f"ledger_{PROFILE}_s{SHARD}.jsonl"),
                reasoning_fields={"reasoning_effort": EFFORT})
by = {t[0]["problem_id"]: t for t in todo}
f = out.open("a")


def save(resp):
    p, history, verdicts = by[resp.key]
    o = resp.output or {}
    f.write(json.dumps({"problem_id": resp.key, "task_name": p["task_name"], "source": p["source"], "family_id": p["family_id"], "set": p.get("set"),
                        "attempt": R, "history_verdicts": verdicts, "feedback_in": history[-1][1], "ok": resp.ok, "model": resp.model,
                        "profile": PROFILE, "effort": EFFORT, "prompt_version": "prompts-1+json_rule kernel_user(history)",
                        "reasoning": o.get("reasoning", ""), "plan": o.get("plan"), "code": o.get("code"),
                        "parsed_from": o.get("_parsed_from", "json"), "text_if_unparsed": o.get("text") if not resp.ok else None,
                        "usage": resp.usage, "error": resp.error}) + "\n")
    f.flush()


k.generate([Request(key=p["problem_id"], system=system, user=kernel_user(ref[p["problem_id"]], history)) for p, history, _ in todo],
           on_result=save)
print(f"REPAIR GEN DONE round {R} {PROFILE} s{SHARD}: spent ${k.ledger.spent:.2f}", flush=True)
