"""Kimi K3.1 round-1 pilot: the exact Claude round-1 prompt (prompts-1 system + docs, render-1 user turn) for each problem in
problems.jsonl, one attempt, through bedrock_kimi.BedrockKimi. Resumable: results stream to out/k31_r1.jsonl, one line per problem
(reasoning, plan, code, usage, error). Spend is capped by the backend's ledger (CAP_USD)."""
import json, os, pathlib, sys
P = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(P))
from bedrock_kimi import BedrockKimi, Request  # noqa: E402

CAP = float(os.environ.get("CAP_USD", "50")); WORKERS = int(os.environ.get("WORKERS", "16")); LIMIT = int(os.environ.get("LIMIT", "0"))
EFFORT = os.environ.get("EFFORT", "high")
rows = [json.loads(l) for l in open(P / "problems.jsonl")]
system = open(rows[0]["prompt"]["system_file"]).read()
assert all(r["prompt"]["system_file"] == rows[0]["prompt"]["system_file"] for r in rows)
out = P / "out" / "k31_r1.jsonl"; out.parent.mkdir(exist_ok=True)
done = {json.loads(l)["problem_id"] for l in open(out)} if out.exists() else set()
todo = [r for r in rows if r["problem_id"] not in done][: LIMIT or None]
print(f"{len(todo)} to generate ({len(done)} done), cap ${CAP}, workers {WORKERS}, effort {EFFORT}", flush=True)
k = BedrockKimi(workers=WORKERS, cap_usd=CAP, max_output_tokens=48000, ledger=str(P / "out" / "ledger.jsonl"),
                reasoning_fields={"reasoning_effort": EFFORT} if EFFORT else None)
by_key = {r["problem_id"]: r for r in todo}
f = out.open("a")

def save(resp):
    o = resp.output or {}
    f.write(json.dumps({"problem_id": resp.key, "task_name": by_key[resp.key]["task_name"], "ok": resp.ok, "model": resp.model,
                        "backend": resp.backend, "effort": EFFORT, "reasoning": o.get("reasoning", ""), "plan": o.get("plan"),
                        "code": o.get("code"), "parsed_from": o.get("_parsed_from", "json"), "text_if_unparsed": o.get("text") if not resp.ok else None,
                        "usage": resp.usage, "error": resp.error}) + "\n"); f.flush()

k.generate([Request(key=r["problem_id"], system=system, user=r["prompt"]["user"]) for r in todo], on_result=save)
print(f"PILOT GEN DONE: spent ${k.ledger.spent:.2f}", flush=True)
