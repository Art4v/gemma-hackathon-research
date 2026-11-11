"""passn.py on the deployed Modal app (gemma-bench, A10G, Ollama 0.40.14: the JAXBench backend) instead of this Mac's Ollama, so val
for every model runs on one backend. Same sampling as passn.py (temperature 1.0, top_p 0.95, top_k 64; seed crc32("<task>/<chain>/0"),
the A2 proposal), the same prompts (diag/prompts_val.jsonl, bench.request_body) and the same three CPU gates (check.py: oracle step 1,
v6e lowering, TPU-interpret numerics). Every sample records the served model and digest; EXPECT (a digest prefix) must match or it stops.
  MODAL_PROFILE=newacct python modal_passn.py TAG N EXPECT   (resumable; samples/modal_<tag>.jsonl)"""
import json, pathlib, re, sys, threading, zlib, concurrent.futures as cf
P = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-hackathon-research/control")
from bench import request_body  # noqa: E402
sys.path.insert(0, str(P))
import modal  # noqa: E402

tag, N, EXPECT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
CODE = re.compile(r"```(?:python)?\n(.*?)```", re.S)  # passn.py's extraction, unchanged
rows = [json.loads(l) for l in open(P.parent / "diag" / "prompts_val.jsonl")]
out = P / "samples" / f"modal_{tag.replace(':', '_')}.jsonl"
done = {(json.loads(l)["task"], json.loads(l)["chain"]) for l in open(out)} if out.exists() else set()


def gate(task, code):  # passn.py's gate, unchanged
    import subprocess
    PY = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python"
    f = P / "samples" / "k" / f"{task}.{abs(hash(code)) % 10**8}.py"; f.parent.mkdir(exist_ok=True); f.write_text(code)
    try:
        p = subprocess.run([PY, "check.py", task, str(f)], cwd=P, capture_output=True, text=True, timeout=900)
        lines = p.stdout.splitlines()
    except subprocess.TimeoutExpired:
        lines = ["TIMEOUT"]
    pick = lambda k: next((l for l in lines if l.startswith(k)), "")
    return {"static": pick("STATIC")[:200], "lower": pick("LOWER")[:200], "numeric": pick("NUMERIC")[:200], "final": lines[-1] if lines else "?"}


jobs, bodies = [], []
for c in range(N):  # chain-major, as passn.py
    for r in rows:
        if (r["task"], c) in done:
            continue
        b = request_body(r["messages"], zlib.crc32(f"{r['task']}/{c}/0".encode()), tag)
        b["options"].update({"temperature": 1.0, "top_p": 0.95, "top_k": 64})
        jobs.append((r["task"], c)); bodies.append(b)
print(f"{len(jobs)} samples to generate for {tag} ({len(done)} done)", flush=True)
gemma = modal.Cls.from_name("gemma-bench", "Gemma")()
pool = cf.ThreadPoolExecutor(3); pending = []; lock = threading.Lock()
with open(out, "a") as fo:
    for (task, c), o in zip(jobs, gemma.chat.map(bodies, order_outputs=True, return_exceptions=True)):
        if isinstance(o, Exception):
            print("ERROR", task, c, str(o)[:200], flush=True); continue
        d = (o.get("backend") or {}).get("digest") or ""
        assert o.get("model") == tag and d.startswith(EXPECT), f"served {o.get('model')} {d[:12]}, expected {tag} {EXPECT}"
        reply = o["message"]["content"]; m = CODE.search(reply or "")
        rec = {"task": task, "chain": c, "tokens": o.get("eval_count"), "finish": o.get("done_reason"), "model": tag, "digest": d[:12],
               "backend": "modal-a10g-ollama-0.40.14", "reply": reply}

        def finish(rec=rec, code=(m.group(1) if m else None)):
            rec.update(gate(rec["task"], code) if code else {"final": "no_code"})
            with lock:
                fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(rec["task"][:40], rec["chain"], rec["tokens"], rec["finish"], rec["final"][:30], flush=True)
        pending.append(pool.submit(finish))
    for f in pending:
        f.result()
print("MODAL_PASSN_DONE", out.name)
