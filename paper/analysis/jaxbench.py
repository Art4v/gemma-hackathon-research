"""Every JAXBench number in the paper, recomputed from the raw run files in paper/data/runs/ (standard library only).

  python3 paper/analysis/jaxbench.py            -> paper/analysis/results.json and paper/analysis/results.md

Definitions (control/PROTOCOL.md, unchanged since the first LoRA run):
  - tasks: JAXBench level 1 (50) + 14 level 2-only tasks with plain-JAX references (64). Primary set = the 50 level 1 tasks + the 8
    level 2 tasks whose reference verified against Google's kernel on the TPU (51p 53p 56p 57p 58p 61p 62p 63p): 58 tasks, 33 k + 25 p.
  - 3 attempts per task: turn 0 + 2 repairs with the oracle's verbatim feedback. A task stops at its first pass (or a terminal harness
    verdict).
  - pass_within_3: a `pass` (correct on 3 seeds and >= 0.7x XLA) on any turn. correct_within_3: `pass` or `correct_slow` on any turn.
    pass_at_1: `pass` at turn 0. parity_within_3: a correct attempt at >= 0.97x XLA.
  - Paired comparison: exact two-sided McNemar on per-task outcomes (binomial on the discordant tasks).
"""
import collections, json, math, pathlib, statistics

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parent / "data" / "runs"
PRIMARY_L2 = {"51p", "53p", "56p", "57p", "58p", "61p", "62p", "63p"}
STEP1_FAIL = {"compile", "not_a_kernel"}
CORRECT = {"pass", "correct_slow"}
# arm label -> (run dir, arm prefix in ids, model label)
ARMS = {
    "bare": ("gemma4_e4b-it-qat__modal", "bare", "Gemma 4 E4B (base), reference only"),
    "docs": ("gemma4_e4b-it-qat__modal", "docs", "Gemma 4 E4B (base) + 25k tokens of Pallas docs"),
    "flash_docs": ("gemini-3.6-flash", "docs", "Gemini Flash + docs"),
    "ours": ("gemma4-e4b-lora__modal", "bare", "Gemma 4 E4B + our LoRA"),
}


def num(task):
    return task.split("_")[0]


def is_primary(task):
    n = int("".join(c for c in num(task) if c.isdigit()))
    return n <= 50 or num(task) in PRIMARY_L2


def kind(task):
    return "k" if num(task).endswith("k") else "p"


def load(run, prefix):
    d = RUNS / run
    gen = [json.loads(l) for f in d.glob(f"{prefix}.jsonl") for l in open(f) if l.strip()]
    V = {}
    for l in open(d / "verdicts.jsonl"):
        if l.strip():
            r = json.loads(l)
            if not r.get("meta") and str(r.get("id", "")).startswith(prefix + "/"):
                V[r["id"]] = r  # last row per id wins (re-ingests)
    tasks = sorted({r["task"] for r in gen})
    out = {}
    for t in tasks:
        turns = []
        for k in range(3):
            v = V.get(f"{prefix}/{t}/t{k}")
            if v:
                turns.append({"turn": k, "verdict": v.get("verdict"), "speed": v.get("speed_vs_xla")})
        out[t] = turns
    return out, len(gen)


def task_outcome(turns):
    p = any(x["verdict"] == "pass" for x in turns)
    c = any(x["verdict"] in CORRECT for x in turns)
    par = any(x["verdict"] in CORRECT and (x["speed"] or 0) >= 0.97 for x in turns)
    p1 = bool(turns) and turns[0]["turn"] == 0 and turns[0]["verdict"] == "pass"
    speeds = [x["speed"] for x in turns if x["verdict"] in CORRECT and x["speed"]]
    return {"pass": p, "correct": c, "parity": par, "pass_at_1": p1, "best_correct_speed": max(speeds) if speeds else None}


def mcnemar(a, b, tasks, key):
    n10 = sum(1 for t in tasks if a[t][key] and not b[t][key])
    n01 = sum(1 for t in tasks if b[t][key] and not a[t][key])
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"a_only": n10, "b_only": n01, "p_exact_two_sided": round(p, 4)}


def main():
    res, md = {"definitions": __doc__.split("Definitions")[1].strip(), "arms": {}}, []
    outcomes, raw = {}, {}
    for arm, (run, prefix, label) in ARMS.items():
        per, ngen = load(run, prefix)
        raw[arm] = per
        oc = {t: task_outcome(v) for t, v in per.items()}
        outcomes[arm] = oc
        prim = [t for t in oc if is_primary(t)]
        def summ(ts):
            cs = [oc[t]["best_correct_speed"] for t in ts if oc[t]["best_correct_speed"]]
            return {"n": len(ts), "pass_within_3": sum(oc[t]["pass"] for t in ts), "correct_within_3": sum(oc[t]["correct"] for t in ts),
                    "pass_at_1": sum(oc[t]["pass_at_1"] for t in ts), "parity_within_3": sum(oc[t]["parity"] for t in ts),
                    "median_best_correct_speed": round(statistics.median(cs), 3) if cs else None,
                    "passed_tasks": sorted(num(t) for t in ts if oc[t]["pass"])}
        funnel = collections.Counter()
        for t, turns in per.items():
            for x in turns:
                funnel[(x["turn"], x["verdict"])] += 1
        res["arms"][arm] = {"label": label, "run_dir": run, "generated_rows": ngen,
                            "all": summ(list(oc)), "primary": summ(prim),
                            "primary_k": summ([t for t in prim if kind(t) == "k"]), "primary_p": summ([t for t in prim if kind(t) == "p"]),
                            "funnel_by_turn": {f"t{k}": dict(sorted(((v or "none"), n) for (kk, v), n in funnel.items() if kk == k)) for k in range(3)}}
    prim_all = sorted(t for t in outcomes["ours"] if is_primary(t))
    res["mcnemar_ours_vs"] = {arm: {"pass_at_1": mcnemar(outcomes["ours"], outcomes[arm], [t for t in prim_all if t in outcomes[arm]], "pass_at_1"),
                                  "pass_within_3": mcnemar(outcomes["ours"], outcomes[arm], [t for t in prim_all if t in outcomes[arm]], "pass"),
                                  "correct_within_3": mcnemar(outcomes["ours"], outcomes[arm], [t for t in prim_all if t in outcomes[arm]], "correct")}
                            for arm in ARMS if arm != "ours"}
    # per-task verdicts of our adapter and turn-to-turn transitions
    res["ours_per_task"] = {t: [(x["turn"], x["verdict"], round(x["speed"], 2) if x["speed"] else None) for x in raw["ours"][t]] for t in sorted(raw["ours"], key=lambda t: int("".join(c for c in num(t) if c.isdigit())))}
    tr = collections.Counter()
    for t, turns in raw["ours"].items():
        for a, b in zip(turns, turns[1:]):
            tr[f"{a['verdict']} -> {b['verdict']}"] += 1
    res["ours_transitions"] = dict(tr.most_common())
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    # markdown tables
    md.append("| Arm | pass within 3 (primary-58) | correct within 3 | pass@1 | parity within 3 | k-33 pass | S-p-25 pass | all-64 pass |")
    md.append("|---|---|---|---|---|---|---|---|")
    for arm, a in res["arms"].items():
        p = a["primary"]
        md.append(f"| {a['label']} | {p['pass_within_3']}/{p['n']} | {p['correct_within_3']}/{p['n']} | {p['pass_at_1']}/{p['n']} | {p['parity_within_3']}/{p['n']} | "
                  f"{a['primary_k']['pass_within_3']}/{a['primary_k']['n']} | {a['primary_p']['pass_within_3']}/{a['primary_p']['n']} | {a['all']['pass_within_3']}/{a['all']['n']} |")
    md.append("\n| ours vs | pass within 3: ours-only / other-only, p | correct within 3: ours-only / other-only, p |\n|---|---|---|")
    for arm, m in res["mcnemar_ours_vs"].items():
        a, c = m["pass_within_3"], m["correct_within_3"]
        md.append(f"| {arm} | {a['a_only']} / {a['b_only']}, p = {a['p_exact_two_sided']} | {c['a_only']} / {c['b_only']}, p = {c['p_exact_two_sided']} |")
    (HERE / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
