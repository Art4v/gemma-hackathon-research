"""Independent recount of the JAXBench table for the LoRA arms, from the committed run rows and the oracle's own files.

  python3 -I recount.py REPO [REF]     # REF: the commit holding control/runs (default origin/hruday/v4-runs)

Per arm it recomputes the headline numbers on the 58-task primary set. For v7 it also checks the rows: one model
digest, the benchmark's sampling settings and seeds, the reference hashes, the turn structure, and that every TPU
verdict in verdicts.jsonl equals the row the TPU VM wrote to its oracle-results branch for the same attempt.
Any failed check raises; the script prints "all checks passed" only at the end.
"""
import collections, hashlib, json, math, subprocess, sys, zlib

REPO = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else "origin/hruday/v4-runs"
ARMS = {"v4": "v4final-e2", "v5": "v5full", "v7": "v7final"}
V7_DIGEST, PROMPT_SHA = "42b1250057a1a8c3", "b186365a"
PRIMARY_L2 = "51p 53p 56p 57p 58p 61p 62p 63p".split()  # control/PROTOCOL.md, "Primary set"
TERMINAL = {"pass", "too_large_for_chip", "harness_oom"}  # bench.py


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True, capture_output=True, text=True).stdout


def jsonl(path):
    return [json.loads(l) for l in git("show", path).splitlines() if l.strip()]


def load(tag):
    d = f"{REF}:control/runs/gemma4-e4b-lora_{tag}__modal"
    return jsonl(f"{d}/bare.jsonl"), {r["id"]: r for r in jsonl(f"{d}/verdicts.jsonl")}  # latest wins, as bench.verdicts


def table(bare, ver, level1):
    by = collections.defaultdict(dict)
    for r in bare:
        by[r["task"]][r["turn"]] = ver[r["id"]]["verdict"]
    primary = [t for t in by if t in level1 or t.split("_")[0] in PRIMARY_L2]
    assert len(primary) == 58, len(primary)
    got = lambda tasks, ok, turns=(0, 1, 2): {t for t in tasks if any(by[t].get(k) in ok for k in turns)}
    return {"pass_within_3": got(primary, ("pass",)), "correct_within_3": got(primary, ("pass", "correct_slow")),
            "pass_turn0": got(primary, ("pass",), (0,)),
            "k33": got([t for t in level1 if t.split("_")[0].endswith("k")], ("pass",)),
            "p25": got([t for t in primary if t.split("_")[0].endswith("p")], ("pass",)),
            "all64": got(list(by), ("pass",)), "tasks": len(by)}


def mcnemar(a, b):
    """Exact two-sided McNemar p for paired task sets a and b."""
    x, y = len(a - b), len(b - a)
    return x, y, min(1.0, 2 * sum(math.comb(x + y, i) for i in range(min(x, y) + 1)) / 2 ** (x + y)) if x + y else 1.0


def check_v7(bare, ver):
    ids = [r["id"] for r in bare]
    assert len(ids) == len(set(ids)) and set(ids) == set(ver), "attempt ids and verdict ids differ"
    digests = collections.Counter(r["backend"]["digest"] for r in bare)
    assert len(digests) == 1 and next(iter(digests)).startswith(V7_DIGEST), digests
    for r in bare:
        assert r["model"] == "gemma4-e4b-lora:v7final" and r["arm"] == "bare", r["id"]
        assert (r["temperature"], r["thinking"], r["has_thinking"], r["num_ctx"], r["max_tokens"]) == (0, False, False, 65536, 8192), r["id"]
        assert r["prompt_sha256"].startswith(PROMPT_SHA), r["id"]
        assert r["seed"] == zlib.crc32(f"{r['task']}/0/{r['turn']}".encode()) & 0x7fffffff, r["id"]  # bench.py's seed
        src = r["ref_source"]
        path = f"benchmarks/{src}/baseline.py" if src.startswith("jaxbench/") else (src if src.endswith(".py") else f"{src}.py")
        assert hashlib.sha256(git("show", f"{REF}:{path}").encode()).hexdigest() == r["ref_sha256"], (r["id"], path)
    turns = collections.defaultdict(dict)
    for r in bare:
        turns[r["task"]][r["turn"]] = ver[r["id"]]["verdict"]
    for t, a in turns.items():
        assert sorted(a) == list(range(len(a))), (t, sorted(a))
        for k, v in a.items():
            assert (k + 1 in a) == (v not in TERMINAL and k < 2), (t, k, v)  # retried exactly when it should be
    tpu = {i for i, v in ver.items() if v.get("device_kind") != "cpu"}
    for i in set(ver) - tpu:  # stopped at oracle step 1 on CPU: a trace-time error, or not a kernel
        v = ver[i]
        assert (v["verdict"], v.get("failed_stage")) in (("compile", "static"), ("not_a_kernel", None)) and v["judged_on"].startswith("cpu: oracle step 1"), i
    code = {r["id"]: r["code"] for r in bare}
    raw, sent = {}, set()
    for k in range(3):
        job = f"control-lora-v7final-modal-t{k}-p1"
        raw.update({r["id"]: r for r in jsonl(f"origin/oracle-results/{job}:oracle/results/{job}/{job}.jsonl") if not r.get("meta")})
        for it in jsonl(f"{REF}:control/runs/cpu_checks/lora-v7final-modal-t{k}_tpu_items.jsonl"):
            assert it["code"] == code[it["id"]], f"{it['id']}: the code sent to the TPU is not the code the model wrote"
            sent.add(it["id"])
    assert set(raw) == tpu == sent, (len(raw), len(tpu), len(sent))
    for i, r in raw.items():
        assert (r["verdict"], r.get("speed_vs_xla"), r["oracle_sha"]) == (ver[i]["verdict"], ver[i].get("speed_vs_xla"), "c9a07d9298a1"), i
    return len(bare), len(tpu), len(turns), dict(collections.Counter(len(a) for a in turns.values()))


if __name__ == "__main__":
    level1 = {p.split("/")[-1] for p in git("ls-tree", "-d", "--name-only", REF, "benchmarks/jaxbench/level1/").split()}
    assert len(level1) == 50
    runs = {a: load(tag) for a, tag in ARMS.items()}
    res = {a: table(*runs[a], level1) for a in ARMS}
    print(f"run rows at {git('rev-parse', '--short', REF).strip()}")
    for a, t in res.items():
        print(f"{a}: pass_within_3 {len(t['pass_within_3'])}/58 | correct within 3 {len(t['correct_within_3'])}/58 | pass at turn 0 "
              f"{len(t['pass_turn0'])}/58 | k-33 {len(t['k33'])}/33 | p-25 {len(t['p25'])}/25 | all tasks run {len(t['all64'])}/{t['tasks']}")
    print("v7 passes:", sorted(t.split("_")[0] for t in res["v7"]["pass_within_3"]))
    for m, other in (("pass_within_3", "v4"), ("pass_within_3", "v5"), ("correct_within_3", "v4")):
        x, y, p = mcnemar(res["v7"][m], res[other][m])
        print(f"McNemar {m}, v7 vs {other}: {x} vs {y} discordant, exact two-sided p = {p:.4f}")
    rows, tpu, tasks, attempts = check_v7(*runs["v7"])
    print(f"v7 rows: {rows} attempts over {tasks} tasks (attempts per task {attempts}), {tpu} judged on the TPU, {rows - tpu} stopped at the CPU kernel check")
    print("all checks passed")
