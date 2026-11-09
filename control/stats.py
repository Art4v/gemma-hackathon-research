"""Statistics for the control benchmark, per PROTOCOL.md: per-arm funnel, pass/correct/parity within k attempts, success vs attempt
budget, Wilson intervals, and paired comparisons between arms (exact McNemar on pass-within-3, paired bootstrap on the difference).
Reads bench.py's run layout read-only; one arm = (model tag, arm name).

  python control/stats.py --arm gemma4_e4b-it-qat__modal:bare --arm gemini-3.6-flash:docs [--pair 0 1] [--tasks-file T] [--json OUT]
"""
import argparse, json, math, random, statistics, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402

STEP1 = {"not_a_kernel"}
STEP2 = {"compile", "vmem", "mosaic", "timeout", "runtime"}
CORRECT = {"pass", "correct_slow"}
HARNESS = {"harness_oom", "too_large_for_chip", "crash"}
PARITY = 0.97


def load_arm(model: str, arm: str) -> dict:
    """{task: [verdict row per turn, in turn order]}; rows without a verdict are skipped."""
    vs, out = bench.verdicts(model), {}
    for r in bench.read_jsonl(bench.run_dir(model) / f"{arm}.jsonl"):
        v = vs.get(r["id"])
        if v is not None:
            out.setdefault(r["task"], []).append(dict(v, turn=r["turn"], prompt_tokens=r.get("prompt_tokens"), output_tokens=r.get("output_tokens")))
    return {t: sorted(x, key=lambda v: v["turn"]) for t, x in out.items()}


def first_index(turns: list, ok):
    return next((i for i, v in enumerate(turns) if ok(v)), None)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def summarize(arm: dict, tasks: list, budget: int = 3) -> dict:
    n = len(tasks)
    def within(ok, k):
        return sum(1 for t in tasks if first_index(arm.get(t, [])[:k], ok) is not None)
    passed = lambda v: v["verdict"] == "pass"
    correct = lambda v: v["verdict"] in CORRECT
    speeds = [v["speed_vs_xla"] for t in tasks for v in arm.get(t, [])[:budget] if v["verdict"] in CORRECT and v.get("speed_vs_xla")]
    best = [max((v["speed_vs_xla"] for v in arm.get(t, [])[:budget] if v["verdict"] in CORRECT and v.get("speed_vs_xla")), default=None) for t in tasks]
    first = [arm[t][0]["verdict"] for t in tasks if arm.get(t)]
    pk = within(passed, budget)
    return {"tasks": n, "judged": sum(1 for t in tasks if arm.get(t)),
            "pass_at_1": within(passed, 1), "pass_within": pk, "pass_within_ci95": wilson(pk, n),
            "correct_within": within(correct, budget), "parity_within": sum(1 for b in best if b is not None and b >= PARITY),
            "success_vs_attempts": [within(passed, k) for k in range(1, budget + 1)],
            "funnel_turn0": {"is_kernel": sum(v not in STEP1 for v in first), "compiles_runs": sum(v not in STEP1 | STEP2 for v in first),
                             "correct": sum(v in CORRECT for v in first), "pass": sum(v == "pass" for v in first),
                             "harness": sum(v in HARNESS for v in first)},
            "median_speed_correct": round(statistics.median(speeds), 3) if speeds else None,
            "prompt_tokens": sum((v.get("prompt_tokens") or 0) for t in tasks for v in arm.get(t, [])[:budget]),
            "output_tokens": sum((v.get("output_tokens") or 0) for t in tasks for v in arm.get(t, [])[:budget])}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b (only A solved) and c (only B solved)."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a: dict, b: dict, tasks: list, budget: int = 3, reps: int = 10000, seed: int = 0) -> dict:
    sa = [int(first_index(a.get(t, [])[:budget], lambda v: v["verdict"] == "pass") is not None) for t in tasks]
    sb = [int(first_index(b.get(t, [])[:budget], lambda v: v["verdict"] == "pass") is not None) for t in tasks]
    only_a, only_b = sum(x and not y for x, y in zip(sa, sb)), sum(y and not x for x, y in zip(sa, sb))
    rng, diffs = random.Random(seed), []
    for _ in range(reps):
        idx = [rng.randrange(len(tasks)) for _ in tasks]
        diffs.append(sum(sa[i] - sb[i] for i in idx) / len(tasks))
    diffs.sort()
    return {"a_pass": sum(sa), "b_pass": sum(sb), "only_a": only_a, "only_b": only_b, "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4),
            "diff": round((sum(sa) - sum(sb)) / len(tasks), 4), "diff_ci95": (round(diffs[int(0.025 * reps)], 4), round(diffs[int(0.975 * reps)], 4))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="MODEL_TAG:ARM, e.g. gemini-3.6-flash:docs")
    ap.add_argument("--pair", nargs=2, type=int, action="append", default=[], help="indices of two --arm values to compare")
    ap.add_argument("--tasks-file", help="one task per line (the primary set); default: all bench tasks")
    ap.add_argument("--budget", type=int, default=3); ap.add_argument("--json")
    a = ap.parse_args(argv)
    tasks = [l.strip() for l in open(a.tasks_file) if l.strip()] if a.tasks_file else list(bench.tasks())
    arms = [load_arm(*s.split(":", 1)) for s in a.arm]
    out = {"tasks": len(tasks), "arms": {s: summarize(x, tasks, a.budget) for s, x in zip(a.arm, arms)},
           "pairs": {f"{a.arm[i]} vs {a.arm[j]}": paired(arms[i], arms[j], tasks, a.budget) for i, j in a.pair}}
    print(json.dumps(out, indent=1))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1))


def _selftest() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(6, 0) - 0.03125) < 1e-9 and abs(mcnemar_exact(5, 0) - 0.0625) < 1e-9
    lo, hi = wilson(8, 64); assert 0.06 < lo < 0.07 and 0.22 < hi < 0.24, (lo, hi)
    arm = {"t1": [{"verdict": "compile", "turn": 0}, {"verdict": "pass", "turn": 1, "speed_vs_xla": 1.2}],
           "t2": [{"verdict": "correct_slow", "turn": 0, "speed_vs_xla": 0.5}], "t3": [{"verdict": "not_a_kernel", "turn": 0}]}
    s = summarize(arm, ["t1", "t2", "t3", "t4"])
    assert (s["pass_at_1"], s["pass_within"], s["correct_within"], s["parity_within"]) == (0, 1, 2, 1), s
    assert s["success_vs_attempts"] == [0, 1, 1] and s["funnel_turn0"]["is_kernel"] == 2 and s["funnel_turn0"]["correct"] == 1, s
    p = paired(arm, {}, ["t1", "t2", "t3", "t4"], reps=200)
    assert (p["only_a"], p["only_b"], p["a_pass"]) == (1, 0, 1), p


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        _selftest(); print("selftest ok")
    else:
        main()
