"""Final verdict per matmul-led k-task for past JAXBench runs, mirroring bench.summary's `best` rule (first pass, else last judged turn).
usage: python3 -I ktable.py RUN_DIR [RUN_DIR ...]"""
import collections, json, pathlib, sys
K = "19 20 21 23 24 26 27 29 30 31 32 35 36 37 38 40 41 42 43 44 45 47 48 49 50".split()
CATS = ("pass", "correct_slow", "vmem", "wrong", "compile")

def final(run):
    rows = [json.loads(l) for l in (run / "bare.jsonl").read_text().splitlines() if l.strip()]
    vs = {}
    for l in (run / "verdicts.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l); vs[r["id"]] = r  # latest wins, as bench.verdicts
    by = collections.defaultdict(list)
    for r in rows:
        if r["id"] in vs:
            by[r["task"]].append(vs[r["id"]] | {"turn": r["turn"]})
    out = {}
    for t, x in by.items():
        x = sorted(x, key=lambda v: v["turn"])
        out[t] = next((v for v in x if v["verdict"] == "pass"), x[-1])
    return out, {r["task"] for r in rows}

def cat(v):
    return v["verdict"] if v["verdict"] in CATS else "other"

if __name__ == "__main__":
    runs = [pathlib.Path(p) for p in sys.argv[1:]]
    res = {r.name: final(r) for r in runs}
    names = {}
    for _, (f, tasks) in res.items():
        for t in tasks:
            n = t.split("_")[0]
            if n.endswith("k") and n[:-1] in K:
                names[n[:-1]] = t
    assert len(names) == 25, sorted(names)
    print("task".ljust(46) + "".join(r.name.split("_")[1][:14].ljust(16) for r in runs))
    for k in K:
        t = names[k]
        cells = []
        for r in runs:
            v = res[r.name][0].get(t)
            cells.append((f"{cat(v)}({v['verdict']},t{v['turn']}" + (f",{v['speed_vs_xla']:.2f}" if v.get("speed_vs_xla") is not None else "") + ")") if v else "UNJUDGED")
        print(t[:45].ljust(46) + "".join(c.ljust(16) if len(c) < 16 else c + " " for c in cells))
    for r in runs:
        f = res[r.name][0]
        c = collections.Counter(cat(f[names[k]]) if names[k] in f else "unjudged" for k in K)
        print(r.name, dict(c), "passed:", [names[k].split("_")[0] for k in K if names[k] in f and f[names[k]]["verdict"] == "pass"])
        print("  all-64 final:", dict(collections.Counter(v["verdict"] for v in f.values())), "judged", len(f))
