"""Kimi K3.1 over the full problem list (full_problems.jsonl), one first attempt each, the constructor's prompts-1 system (+docs) and the
render-1 user turn. Sharded across Bedrock inference profiles (each has its own tokens-per-minute quota):
  PROFILE=global|us SHARD=i NSHARDS=n CAP_USD=... python run_full.py
Resumable: skips every problem_id already in any out/k31_*.jsonl with ok=true. Spend capped per shard by the backend ledger."""
import glob, json, os, pathlib, sys, zlib
P = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(P))
from bedrock_kimi import BedrockKimi, Request  # noqa: E402

PROFILE, SHARD, NSH = os.environ["PROFILE"], int(os.environ["SHARD"]), int(os.environ["NSHARDS"])
CAP, WORKERS, EFFORT = float(os.environ["CAP_USD"]), int(os.environ.get("WORKERS", "12")), os.environ.get("EFFORT", "high")
PRICES = {"global": (4.0, 20.0), "us": (4.4, 22.0)}[PROFILE]
system = open(P / "system_prompts-1_with_docs.txt").read()
PROBLEMS = pathlib.Path(os.environ.get("PROBLEMS", str(P / "full_problems.jsonl")))
OUTD = pathlib.Path(os.environ.get("OUTDIR", str(P / "out"))); TAG = os.environ.get("TAG", "full"); OUTD.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(PROBLEMS)]
rows = [r for r in rows if zlib.crc32(r["problem_id"].encode()) % NSH == SHARD]
done = set()
for f in glob.glob(str(OUTD / "k31_*.jsonl")):
    for l in open(f):
        d = json.loads(l)
        if d.get("ok"):
            done.add(d["problem_id"])
todo = [r for r in rows if r["problem_id"] not in done]
out = OUTD / f"k31_{TAG}_{PROFILE}_s{SHARD}.jsonl"
print(f"{PROFILE} shard {SHARD}/{NSH}: {len(todo)} to generate ({len(rows) - len(todo)} done), cap ${CAP}, workers {WORKERS}", flush=True)
k = BedrockKimi(model=f"{PROFILE}.moonshotai.kimi-k3.1", workers=WORKERS, cap_usd=CAP, max_output_tokens=48000,
                price_in=PRICES[0], price_out=PRICES[1], ledger=str(OUTD / f"ledger_{TAG}_{PROFILE}_s{SHARD}.jsonl"),
                reasoning_fields={"reasoning_effort": EFFORT})
by_key = {r["problem_id"]: r for r in todo}
f = out.open("a")

def save(resp):
    o = resp.output or {}
    r = by_key[resp.key]
    f.write(json.dumps({"problem_id": resp.key, "task_name": r["task_name"], "source": r["source"], "family_id": r["family_id"], "set": r.get("set"),
                        "ok": resp.ok, "model": resp.model, "profile": PROFILE, "effort": EFFORT, "system_sha256": "dc7c4acc72f1",
                        "prompt_version": "prompts-1+json_rule", "render_version": "render-1", "reasoning": o.get("reasoning", ""),
                        "plan": o.get("plan"), "code": o.get("code"), "parsed_from": o.get("_parsed_from", "json"),
                        "text_if_unparsed": o.get("text") if not resp.ok else None, "usage": resp.usage, "error": resp.error}) + "\n")
    f.flush()

k.generate([Request(key=r["problem_id"], system=system, user=r["user"]) for r in todo], on_result=save)
print(f"FULL GEN DONE {PROFILE} s{SHARD}: spent ${k.ledger.spent:.2f}", flush=True)
