"""Collect every number the figures use into data.json, read straight from the commits that hold the evidence.

  python3 paper/figures/src/extract.py [REPO]      # REPO: a clone with the refs below fetched (default: this checkout)

Nothing is typed in by hand except DEV_LOSS, OURS_PRIMARY and EXPECT. Each count is asserted against those, so a drifted source fails here
instead of producing a wrong figure.
"""
import collections, csv, difflib, io, json, pathlib, re, subprocess, sys

REPO = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).resolve().parents[3])
TRAIN = "2efa6a5:kimi/v7/v7_final.jsonl"        # hruday/kimi-k31: the file v7 was trained on
RES = "6e898c7"                                # TPU verdicts for the v7 training candidates (tag archive/oracle-results/karthik-gen7)
RUNS = "487973d"                                # hruday/v4-runs: the scored run folder and the 64 references
LOSS = "eadf22b:paper/parts/loss_v7.csv"         # aarav/paper-v7: one row per example step of the v7 training run
DEV_LOSS = [[0, 1.318], [100, 0.521], [300, 0.018], [500, 0.005], [4608, 0.0002]]  # (example step, loss): training_v7.md @ eadf22b
PRIMARY_L2 = "51p 53p 56p 57p 58p 61p 62p 63p".split()      # control/PROTOCOL.md, "Primary set"
MATMUL_K = "19 20 21 23 24 26 27 29 30 31 32 35 36 37 38 40 41 42 43 44 45 47 48 49 50".split()
PAPER = "da04f91"                               # hruday/paper-v7: paper/data/runs (the frozen rows of the four arms) and paper/analysis/results.json
LORA_RUN = "gemma4-e4b-lora_v7final__modal"     # the scored adapter's run folder at RUNS
ARMS = {"bare": ("Gemma 4 E4B", "gemma4_e4b-it-qat__modal", "bare"), "docs": ("Gemma 4 E4B + docs", "gemma4_e4b-it-qat__modal", "docs"),
        "flash_docs": ("Gemini 3.6 Flash + docs", "gemini-3.6-flash", "docs"),
        "ours": ("Gemma 4 E4B + our LoRA", "gemma4-e4b-lora__modal", "bare")}  # results.json key: (label, run folder, arm file)
OURS_PRIMARY = {"b": 24, "c": 21, "d": 9, "e": 9}  # recounted by oracle/tools/v7_audit/recount.py on oracle-results/karthik-gen7
EXPECT = {"kimi_r3": (126, 41), "gen7": (281, 269), "gen7L": (111, 104), "gen7R": (61, 61), "gen7V": (91, 91)}  # (judged, pass)
HALT = ("halted unexpectedly", "RuntimeUnexpectedCoreHalt")
BEFORE_REF, AFTER_REF = "Here is the reference JAX function:", "Please convert this function into a Pallas TPU kernel."  # bench.py


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], check=True, capture_output=True, text=True).stdout


def jsonl(ref_path):
    return [json.loads(l) for l in git("show", ref_path).splitlines() if l.strip()]


def verdicts(ref, paths):
    """id -> verdict row over several result files, later files winning. A row written while the TPU core was halted
    is replaced by any real verdict for the same id (a victim that was re-run); a kernel that halts the core on every
    run has only such rows and keeps one."""
    out, halted = {}, {}
    for p in paths:
        for r in jsonl(f"{ref}:{p}"):
            if r.get("meta"):
                continue
            is_halt = r.get("verdict") == "harness_oom" and any(h in (r.get("feedback") or "") for h in HALT)
            (halted if is_halt else out)[r["id"]] = r
    return halted | out


def cls(v):
    return {"mosaic": "compile"}.get(v, v if v in ("pass", "correct_slow", "wrong", "vmem", "compile") else "other")


def train(rows):
    count = lambda f: dict(collections.Counter(f(r) for r in rows))
    gen = [r for r in rows if r["teacher"] == "generator-v7"]
    out = {"rows": len(rows), "by_set": count(lambda r: r["set"]), "by_teacher": count(lambda r: r["teacher"]),
           "by_kind": count(lambda r: r["kind"]), "by_source": count(lambda r: r["source"]), "by_verdict": count(lambda r: r["verdict"]),
           "by_messages": count(lambda r: str(len(r["messages"]))), "by_family": dict(collections.Counter(r["family"] for r in gen)),
           "variants": sum(1 for r in gen if is_variant(r)),
           "speed": {t: sorted(round(r["speed_vs_xla"], 4) for r in rows if r["teacher"] == t) for t in ("kimi-k3.1", "generator-v7")}}
    assert out["rows"] == 2304 and out["by_teacher"] == {"kimi-k3.1": 1595, "generator-v7": 709}, out["by_teacher"]
    assert out["variants"] == 354, out["variants"]
    return out


def is_variant(r):
    return bool(r.get("surface_of") or r.get("surface_of_item"))


def candidates():
    ls = lambda d: [p for p in git("ls-tree", "-r", "--name-only", RES, f"oracle/results/{d}/").split() if p.endswith(".jsonl")]
    files = {"kimi_r3": ls("kimi-r3"), "gen7": ["oracle/results/gen7/gen7-full2.merged.jsonl"],
             "gen7L": ls("gen7L"), "gen7R": [p for p in ls("gen7R") if re.search(r"gen7R-[ab]\.jsonl$", p)],
             "gen7V": [p for p in ls("gen7V") if "items" not in p]}
    out = {}
    for k, paths in files.items():
        c = collections.Counter(cls(v["verdict"]) for v in verdicts(RES, sorted(paths)).values())
        assert (sum(c.values()), c["pass"]) == EXPECT[k], (k, dict(c))
        out[k] = dict(c)
    return out


def sweep():
    out = []
    for n in (1, 2):
        for r in verdicts(RES, [f"oracle/results/gen7/gen7-sweep{n}.jsonl"]).values():
            m = re.search(r"matmul_epilogue__(\d+)__t(\d+)_(\d+)_(\d+)$", r["id"])  # tm, tn, tk in the generator's order
            if m:
                out.append({"item": m.group(1), "m": r["shapes"][0][0], "tiles": [int(g) for g in m.groups()[1:]],
                            "verdict": r["verdict"], "speed": round(r["speed_vs_xla"], 4)})
    return out


def exam_refs():
    out = {}
    for p in git("ls-tree", "-r", "--name-only", RUNS, "benchmarks/jaxbench/level1/", "control/refs/").split():
        if p.endswith("/baseline.py") or (p.startswith("control/refs/") and p.endswith(".py")):
            out[p.split("/")[-2] if p.endswith("/baseline.py") else p.split("/")[-1][:-3]] = git("show", f"{RUNS}:{p}")
    assert len(out) == 64, len(out)
    return out


def similarity(rows):
    """How close each training prompt's reference is to its closest benchmark reference: difflib's ratio over stripped,
    non-blank lines (2 x matching lines / total lines, in order). The same measure between each benchmark reference and
    its closest other benchmark reference gives the scale."""
    lines = lambda s: [l.strip() for l in s.splitlines() if l.strip()]
    ratio = lambda a, b: difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    exam = {t: lines(s) for t, s in exam_refs().items()}
    cache, out = {}, []
    for r in rows:
        ref = next(m["content"] for m in r["messages"] if m["role"] == "user").split(BEFORE_REF, 1)[1].split(AFTER_REF, 1)[0]
        ref = re.sub(r"^\s*```(?:python)?\n|\n```\s*$", "", ref.strip())
        assert "def workload" in ref, r["example_id"]
        if ref not in cache:
            cache[ref] = max((ratio(lines(ref), e), t) for t, e in exam.items())
        group = "kimi" if r["teacher"] == "kimi-k3.1" else "gen_variant" if is_variant(r) else "gen"
        out.append({"group": group, "sim": round(cache[ref][0], 4), "nearest": cache[ref][1]})
    between = sorted(round(max(ratio(a, b) for u, b in exam.items() if u != t), 4) for t, a in exam.items())
    return {"rows": out, "between_benchmark_tasks": between, "unique_references": len(cache)}


def loss(rows):
    """Per-example training loss, with each step marked as a teacher row or a generated row through its task."""
    teacher = {}
    for r in rows:
        assert teacher.setdefault(r["task"], r["teacher"]) == r["teacher"], r["task"]
    log = list(csv.DictReader(io.StringIO(git("show", LOSS))))
    for epoch in ("1", "2"):  # each epoch visits exactly the training file
        assert collections.Counter(r["task"] for r in log if r["epoch"] == epoch) == collections.Counter(r["task"] for r in rows), epoch
    return {"loss": [float(r["loss"]) for r in log], "lr": [float(r["lr"]) for r in log],
            "generated": [int(teacher[r["task"]] == "generator-v7") for r in log], "dev_loss": DEV_LOSS}


def results():
    """Every attempt of the four arms, from the paper's frozen rows. The adapter's rows there must be byte-identical to the run
    folder they were scored in, and each arm must agree with the paper's own results.json."""
    level1 = sorted(p.split("/")[-1] for p in git("ls-tree", "-d", "--name-only", RUNS, "benchmarks/jaxbench/level1/").split())
    assert len(level1) == 50, len(level1)
    primary = lambda t: t in level1 or t.split("_")[0] in PRIMARY_L2
    for f in ("bare.jsonl", "verdicts.jsonl"):
        assert git("rev-parse", f"{PAPER}:paper/data/runs/{ARMS['ours'][1]}/{f}") == git("rev-parse", f"{RUNS}:control/runs/{LORA_RUN}/{f}"), f
    published = json.loads(git("show", f"{PAPER}:paper/analysis/results.json"))["arms"]
    arms = {}
    for key, (label, d, arm) in ARMS.items():
        ver = {r["id"]: r for r in jsonl(f"{PAPER}:paper/data/runs/{d}/verdicts.jsonl")}  # latest wins, as bench.verdicts
        by = collections.defaultdict(dict)
        for r in jsonl(f"{PAPER}:paper/data/runs/{d}/{arm}.jsonl"):
            v = ver[r["id"]]
            past_step1 = v["verdict"] != "not_a_kernel" and v.get("failed_stage") != "static"  # got as far as compiling on the TPU
            by[r["task"]][str(r["turn"])] = {"verdict": v["verdict"], "tpu": past_step1, "speed": v.get("speed_vs_xla") and round(v["speed_vs_xla"], 3)}
        assert len(by) == 64 and sum(map(primary, by)) == 58, (key, len(by))
        any_ = lambda ok: sum(1 for t, a in by.items() if primary(t) and any(ok(x) for x in a.values()))
        funnel = {"b": any_(lambda x: x["tpu"]), "c": any_(lambda x: x["verdict"] in ("wrong", "correct_slow", "pass")),
                  "d": any_(lambda x: x["verdict"] in ("correct_slow", "pass")), "e": any_(lambda x: x["verdict"] == "pass")}
        within = [sum(1 for a in by.values() if any(a.get(str(t), {}).get("verdict") == "pass" for t in range(k + 1))) for k in range(3)]
        pub = published[key]
        assert (within[0], within[2], funnel["e"], funnel["d"]) == (pub["all"]["pass_at_1"], pub["all"]["pass_within_3"],
                                                                 pub["primary"]["pass_within_3"], pub["primary"]["correct_within_3"]), key
        arms[key] = {"label": label, "funnel_primary58": funnel, "within_k_of_64": within, "tasks": by}
    assert arms["ours"]["funnel_primary58"] == OURS_PRIMARY, arms["ours"]["funnel_primary58"]
    num = lambda t: (int(re.match(r"\d+", t).group()), t)
    is_mk = lambda t: t.split("_")[0].endswith("k") and t.split("_")[0][:-1] in MATMUL_K
    tasks = sorted(arms["ours"]["tasks"], key=num)
    assert sum(map(is_mk, tasks)) == 25
    return {"matmul_k_tasks": [t for t in tasks if is_mk(t)], "other_tasks": [t for t in tasks if not is_mk(t)], "arms": arms}


if __name__ == "__main__":
    rows = jsonl(TRAIN)
    data = {"sources": {"train": TRAIN, "loss": LOSS, "paper_snapshot": PAPER, "candidates": f"{RES}@{git('rev-parse', '--short', RES).strip()}", "runs": RUNS},
            "train": train(rows), "candidates": candidates(), "sweep": sweep(), "similarity": similarity(rows), "loss": loss(rows), "results": results()}
    out = pathlib.Path(__file__).resolve().parent / "data.json"
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    for m, r in data["results"]["arms"].items():
        print(m, "primary-58 funnel", r["funnel_primary58"], "| within k of 64", r["within_k_of_64"])
    s = data["similarity"]
    for g in ("kimi", "gen", "gen_variant"):
        v = sorted(x["sim"] for x in s["rows"] if x["group"] == g)
        print(f"similarity {g}: n {len(v)}, median {v[len(v) // 2]}, max {v[-1]}")
    b = s["between_benchmark_tasks"]
    print(f"between benchmark tasks: n {len(b)}, min {b[0]}, median {b[len(b) // 2]}, max {b[-1]} | unique references {s['unique_references']}")
    print("wrote", out.name, out.stat().st_size, "bytes")
