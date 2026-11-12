"""v7 Kimi round 3 on the v4-covered gap (r3/gap.py -> r3/gap.json): old-pool problems whose task v4 trained on but v6 has no row for,
whose judged attempt chain (a1 = kimi31r1, a2 = kimi31a2) ends in a kernel-attributable verdict. One repair turn each, exactly as the
pool's repair rounds: kernel_user(ref, [(a1, fb1), (a2, fb2)]), same system prompt (no tiling policy: these are not exam scale).
  PROFILE=global|us SHARD=i NSHARDS=n CAP_USD=... [WORKERS=10] python run_r3.py      (resumable; out/r3/)"""
import json, os, pathlib, sys, zlib
V7 = pathlib.Path(__file__).resolve().parent
K = pathlib.Path.home() / "Projects/gemma-work/wt-kimi/kimi/scripts"
sys.path.insert(0, str(K))
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from bedrock_kimi import BedrockKimi, Request  # noqa: E402
from constructor.prompts import kernel_user  # noqa: E402

PROFILE, SHARD, NSH = os.environ["PROFILE"], int(os.environ["SHARD"]), int(os.environ["NSHARDS"])
CAP, WORKERS, TAG = float(os.environ["CAP_USD"]), int(os.environ.get("WORKERS", "10")), os.environ.get("TAG", "")
PRICES = {"global": (4.0, 20.0), "us": (4.4, 22.0)}[PROFILE]
REPAIR = {"compile", "not_a_kernel", "mosaic", "vmem", "runtime", "wrong", "timeout"}
gap = json.load(open(V7 / "r3/gap.json"))
SALT = os.environ.get("SALT", "")  # top-ups: re-shard on a salted hash; MODS="4:1,3" keeps only pids with crc32 % 4 in {1, 3}
MODS = os.environ.get("MODS")
keep = (lambda pid: zlib.crc32(pid.encode()) % int(MODS.split(":")[0]) in {int(x) for x in MODS.split(":")[1].split(",")}) if MODS else (lambda pid: True)
todo = {pid: g for pid, g in gap.items() if g["chain"] and g["chain"][-1][1] in REPAIR and keep(pid) and zlib.crc32((SALT + pid).encode()) % NSH == SHARD}
outdir = V7 / "out/r3"; outdir.mkdir(parents=True, exist_ok=True)
done = set()
for f in outdir.glob("k31_r3_*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if d.get("ok"):
            done.add(d["problem_id"])
todo = {k: v for k, v in todo.items() if k not in done}
system = open(K / "system_prompts-1_with_docs.txt").read()
print(f"v7 round 3 (v4-covered gap), {PROFILE} shard {SHARD}/{NSH}: {len(todo)} to generate ({len(done)} done), cap ${CAP}", flush=True)
k = BedrockKimi(model=f"{PROFILE}.moonshotai.kimi-k3.1", workers=WORKERS, cap_usd=CAP, max_output_tokens=48000, price_in=PRICES[0], price_out=PRICES[1],
                ledger=str(outdir / f"ledger_{PROFILE}_s{SHARD}{TAG}.jsonl"), reasoning_fields={"reasoning_effort": "high"})
f = (outdir / f"k31_r3_{PROFILE}_s{SHARD}{TAG}.jsonl").open("a")


def save(resp):
    g = todo[resp.key]; o = resp.output or {}
    f.write(json.dumps({"problem_id": resp.key, "task": g["task"], "source": g["source"], "family_id": g["family_id"], "attempt": len(g["chain"]) + 1,
                        "history_verdicts": [c[1] for c in g["chain"]], "ok": resp.ok, "model": resp.model, "profile": PROFILE, "effort": "high",
                        "prompt_version": "prompts-1+json_rule kernel_user(history)", "reasoning": o.get("reasoning", ""), "plan": o.get("plan"),
                        "code": o.get("code"), "parsed_from": o.get("_parsed_from", "json"), "text_if_unparsed": o.get("text") if not resp.ok else None,
                        "usage": resp.usage, "error": resp.error}) + "\n")
    f.flush()


k.generate([Request(key=pid, system=system, user=kernel_user(g["ref"], [(c[0], c[2]) for c in g["chain"]])) for pid, g in todo.items()],
           on_result=save)
print(f"R3 GEN DONE {PROFILE} s{SHARD}: spent ${k.ledger.spent:.2f}", flush=True)
