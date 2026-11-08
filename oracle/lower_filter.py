"""A CPU reject-only pre-filter for kernels, and its validation against kernels the chip has already judged (clanker, issue #1 6713940244).

  python oracle/lower_filter.py validate --py <JAX 0.10.2 python> --out REPORT.json [--workers 3] [--limit N] [--timeout 180]

The filter cross-lowers a candidate for TPU v6e on a CPU host (lower_vmem.py's method: JAX_PLATFORMS=cpu, v6e chip info registered for the CPU
device, jax.jit(workload).trace(*abstract).lower(lowering_platforms=("tpu",))). Pallas's TPU lowering enforces the block-shape rules and
much else, and a trace catches broadcast and API errors, all without a chip. It can only REJECT, never pass: a rejected kernel is reported
and not run; anything else goes to the chip as before, which stays the only judge of a pass. If the reference itself can't be loaded or
shaped, the filter ABSTAINS (the kernel goes to the chip): a harness problem is never the kernel's rejection.

Kernels are model-written, so each is lowered in its own sandboxed `python -I` child, by sandbox/sandbox_profile.py's profile (no network, no exec, no
reads under $HOME but the interpreter, an environment of PATH, JAX_PLATFORMS and a temp HOME). The child prints one JSON result; its error
text is cut to the exception type plus a scrubbed message.

Validation set, every kernel the chip has judged whose code is in the repository:
  - round2/labels/train_r2_v6e_labels.jsonl: the H2 training kernels on a real v6e (ran, or crashed), code from round2/data/train_r2.jsonl,
    abstract inputs at the label's own full-size `shapes`, each with the dtype the row's own reference gives it (if the reference won't
    load or its input count differs from the label's, the filter abstains)
  - oracle/results/*: the study's verdicts, code and reference from oracle/queue/*.json
"Ran on the chip" is pass, correct_slow or wrong (a label: ran). "Failed on the chip" is compile, vmem, mosaic or timeout (a label: not ran).
not_a_kernel and harness_oom say nothing about lowering and are left out. The bar (clanker, 6703335547 and 6713940244): the filter rejects
NONE of the kernels that ran. One wrong rejection and it's out.
"""
import argparse, collections, concurrent.futures as cf, glob, json, os, pathlib, re, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent  # the oracle folder
CHILD_FLAG = "--lower-one"
RAN = {"pass", "correct_slow", "wrong"}
FAILED = {"compile", "vmem", "mosaic", "timeout"}
TIMEOUT = 180


# ---- child: runs inside the sandbox, imports nothing from the repository

def _load(src, name):
    import types
    mod = types.ModuleType(name)
    sys.modules[name] = mod  # dataclasses look their module up here
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    return mod


CHILD_KEEP = 1000  # characters the child keeps from each end of a message: lowering errors put the numbers (the shapes) at the end


def _bound(msg, keep=CHILD_KEEP):
    """Head and tail of a long message, never only the head: "...divisible by 8 and 128 ... the block shape is (4, 100)" ends in the part
    a teacher needs."""
    return msg if len(msg) <= 2 * keep else msg[:keep] + " ... " + msg[-keep:]


def child():
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)  # the candidate's own prints go to stderr, which the parent drops
    stage, result = "setup", None
    try:
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax, jax.numpy as jnp
        from jax._src.pallas.mosaic import tpu_info
        info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion("v6e"), 1)
        tpu_info.registry["cpu"] = lambda: info
        stage = "ref"
        if payload.get("shapes") is None:
            ref = _load(payload["ref"], "lf_ref")
            abstract = jax.eval_shape(ref.create_inputs)
        else:  # a label: full-size shapes from the label, dtypes from its own (capped) reference, leaf by leaf
            ref = _load(payload["ref"], "lf_ref")
            leaves = jax.tree_util.tree_leaves(jax.eval_shape(ref.create_inputs))
            if len(leaves) != len(payload["shapes"]):
                raise ValueError(f"reference has {len(leaves)} inputs, the label {len(payload['shapes'])}")
            abstract = [jax.ShapeDtypeStruct(tuple(sh), x.dtype) for sh, x in zip(payload["shapes"], leaves)]
        abstract = list(abstract) if isinstance(abstract, (tuple, list)) else [abstract]
        stage = "cand"
        cand = _load(payload["cand"], "lf_cand")
        stage = "lower"
        jax.jit(cand.workload).trace(*abstract).lower(lowering_platforms=("tpu",))
        result = {"lowered": True}
    except BaseException as e:  # noqa: BLE001  SystemExit from module top level included
        if isinstance(e, KeyboardInterrupt):
            raise
        result = {"lowered": False, "stage": stage, "error": _bound(f"{type(e).__name__}: {e}")}
    out.write(json.dumps(result) + "\n")
    out.flush()
    os._exit(0)


# ---- parent

def outcome(res):
    """'lowered', 'rejected' (the candidate failed to load or lower) or 'abstain' (setup or reference failed, or no usable result)."""
    if res.get("lowered") is True:
        return "lowered"
    if res.get("stage") in ("cand", "lower"):
        return "rejected"
    return "abstain"


def child_result(stdout, returncode):
    """Accept only the child's own result shape; anything else abstains."""
    try:
        r = json.loads(stdout)
    except ValueError:
        return {"lowered": False, "stage": "child", "error": f"SandboxChild: exit {returncode}, no result"}
    if isinstance(r, dict) and r.get("lowered") is True:
        return {"lowered": True}
    if isinstance(r, dict) and r.get("lowered") is False and r.get("stage") in ("setup", "ref", "cand", "lower") and isinstance(r.get("error"), str):
        return r
    return {"lowered": False, "stage": "child", "error": "SandboxChild: malformed result"}


def lower_sandboxed(payload, py, timeout=TIMEOUT, home=None):
    """One candidate, in its own sandboxed child, by sandbox/sandbox_profile.py's profile. Never runs anything in this process."""
    sys.path.insert(0, str(HERE / "sandbox"))
    import sandbox_profile as screen
    screen._require_sandbox()
    home = os.path.realpath(home or os.path.expanduser("~"))
    with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
        d = os.path.realpath(d)
        child_py = pathlib.Path(d, "lower_filter.py")
        child_py.write_text(pathlib.Path(__file__).read_text())  # the child reads nothing of the repository
        cmd = [screen.SANDBOX_EXEC, "-p", screen._profile(home, d, py), py, "-I", str(child_py), CHILD_FLAG]
        try:
            p = subprocess.run(cmd, input=json.dumps(payload).encode("utf-8"), capture_output=True, timeout=timeout, cwd=d,
                               env={"PATH": screen.MIN_PATH, "JAX_PLATFORMS": "cpu", "HOME": d})
        except subprocess.TimeoutExpired:
            return {"lowered": False, "stage": "child", "error": f"Timeout: killed after {timeout} s"}
    return _post(child_result(p.stdout, p.returncode))


def _post(r):
    """The child's result made safe in two forms (clanker, issue #1 6716525540). error: sandbox_profile's scrub, about 200 characters, for
    outcomes, reports and the validation. message: the child's bounded text with only the key and token patterns redacted, no length cut,
    for feedback_text (which also masks paths and trims to FEEDBACK_CHARS, keeping head and tail). scrub's cut used to reach the feedback
    first, so block-shape messages lost their shapes."""
    if "error" not in r:
        return r
    sys.path.insert(0, str(HERE / "sandbox"))
    import sandbox_profile as screen
    msg = r["error"]
    for rx in screen.SECRETISH:
        msg = rx.sub("<redacted>", msg)
    return {**r, "message": msg, "error": screen.scrub(r["error"])}


def strip_fence(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
    return s[:-3] if s.endswith("```") else s


def label_items(labels, train):
    """(id, payload, chip) for the v6e labels: code from the training row; inputs at the label's full-size shapes, with each input's dtype from
    the training row's own reference (an index input is an integer: a float32 guess there was a harness bug, 2 false rejections)."""
    code, refs = {}, {}
    for t in train:
        code[t["key"]] = strip_fence(t["messages"][-1]["content"])
        m = re.search(r"```python\n(.*?)```", t["messages"][1]["content"], re.S)
        if m:
            refs[t["key"]] = m.group(1)
    out = []
    for r in labels:
        if "meta" in r or r.get("ran") is None or not r.get("shapes"):
            continue
        key = r["id"].split("/", 1)[1] if "/" in r["id"] else r["id"]
        if key in code and key in refs:
            out.append((r["id"], {"cand": code[key], "ref": refs[key], "shapes": r["shapes"]}, "ran" if r["ran"] else "failed"))
    return out


def study_items(results, queue_items):
    """(id, payload, chip) for the study's verdicts whose code and reference are in a queue request."""
    q = {it["id"]: it for it in queue_items}
    out, seen = [], set()
    for r in results:
        v = r.get("verdict")
        chip = "ran" if v in RAN else "failed" if v in FAILED else None
        if chip is None or r["id"] not in q or r["id"] in seen or not q[r["id"]].get("code") or not q[r["id"]].get("ref_code"):
            continue
        seen.add(r["id"])
        out.append((r["id"], {"cand": q[r["id"]]["code"], "ref": q[r["id"]]["ref_code"]}, chip))
    return out


def summarise(rows):
    """rows: [(id, chip, result)]. The bar holds iff no kernel that ran on the chip was rejected."""
    table = collections.Counter((chip, outcome(res)) for _, chip, res in rows)
    wrong = [{"id": i, "error": res.get("error", "")[:200]} for i, chip, res in rows if chip == "ran" and outcome(res) == "rejected"]
    failed = sum(1 for _, chip, _ in rows if chip == "failed")
    caught = table[("failed", "rejected")]
    return {"kernels": len(rows), "table": {f"{c}/{o}": n for (c, o), n in sorted(table.items())},
            "wrongly_rejected": len(wrong), "wrongly_rejected_ids": wrong,
            "chip_failures_caught": caught, "chip_failures": failed, "catch_rate": round(caught / failed, 3) if failed else None,
            "bar_met": not wrong}


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def validation_set(root=HERE):
    labels = read_jsonl(root / "round2/labels/train_r2_v6e_labels.jsonl")
    train = read_jsonl(root / "round2/data/train_r2.jsonl")
    results = [r for f in sorted(glob.glob(str(root / "results/*/*.jsonl"))) for r in read_jsonl(f)]
    queue = [it for f in sorted(glob.glob(str(root / "queue/*.json"))) for it in json.load(open(f)).get("items", [])]
    return label_items(labels, train) + study_items(results, queue)


def validate(items, run, workers=3):
    """items: [(id, payload, chip)]; run(payload) -> child result. Returns (rows, summary)."""
    with cf.ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda it: run(it[1]), items))
    rows = [(i, chip, res) for (i, _, chip), res in zip(items, results)]
    return rows, summarise(rows)


def filter_rows(items, rows, run, workers=3, with_results=False):
    """A generate run's kernels, before packing (clanker, issue #1 6715359660): [{id, task, outcome, stage, error_type}] for every row with code.
    run(payload) -> child result. Only the exception type is kept here; feedback_rows carries the redacted message for the repair turn.
    with_results: also return (the rows judged, the child results), for feedback_rows."""
    refs = {it["task"]: it["ref"] for it in items}
    todo = [r for r in rows if r.get("code") and r["task"] in refs]
    with cf.ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda r: run({"cand": r["code"], "ref": refs[r["task"]]}), todo))
    out = [{"id": r["id"], "task": r["task"], "outcome": outcome(res), "stage": None if res.get("lowered") is True else res.get("stage"),
            "error_type": (res.get("error") or "").split(":")[0] or None} for r, res in zip(todo, results)]
    return (out, todo, results) if with_results else out


FEEDBACK_CHARS = 400  # the oracle's own bound (tpu_oracle.FEEDBACK_CHARS): a block-shape message gives its numbers after ~200 characters
_PATH = re.compile(r"(?:/(?:Users|private|home|var|tmp|opt|usr|Library)/|[A-Za-z]:\\)[^\s'\"`,;)\]}]*")


def feedback_text(error):
    """The lowering message as a teacher can use it, safe to commit (clanker, issue #1 6715692985): key- or token-like text redacted with
    sandbox_profile's own patterns, absolute paths (this machine's user name, the sandbox's temp dir, the venv) masked, and at most
    FEEDBACK_CHARS, keeping the head (what failed) and the tail (the numbers)."""
    sys.path.insert(0, str(HERE / "sandbox"))
    import sandbox_profile as screen
    kind, _, msg = (error or "").partition(": ")
    for rx in screen.SECRETISH:
        msg = rx.sub("<redacted>", msg)
    msg = " ".join(_PATH.sub("<path>", msg).split())
    if len(msg) > FEEDBACK_CHARS:
        msg = msg[: FEEDBACK_CHARS // 3] + " ... " + msg[-(FEEDBACK_CHARS - FEEDBACK_CHARS // 3):]
    return f"compile error (TPU lowering, checked on a CPU host): {kind}: {msg}" if kind else "compile error (TPU lowering, checked on a CPU host)"


def feedback_rows(rows, results):
    """A results-style row for every rejected kernel, which `rounds.py generate --results` reads as a verdict: verdict "compile" isn't
    terminal, so `pending` sends it back through `repair_turn` with this feedback (clanker's contract test, tests/test_rounds.py). A kernel
    that lowered gets no row: it waits for the chip. judge says it was the CPU lowering, never the chip."""
    return [{"id": r["id"], "task": r["task"], "verdict": "compile", "judge": "cpu_lowering",
             "feedback": feedback_text(res.get("message") or res.get("error"))} for r, res in zip(rows, results) if outcome(res) == "rejected"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--py", required=True, help="a JAX 0.10.2 interpreter (the oracle's pin)")
    v.add_argument("--out", required=True); v.add_argument("--workers", type=int, default=3)  # about 300 MB each: mind a 16 GB machine
    v.add_argument("--limit", type=int, default=0); v.add_argument("--timeout", type=int, default=TIMEOUT)
    f = sub.add_parser("filter", help="lower a generate run's kernels before packing; reject-only")
    f.add_argument("--py", required=True); f.add_argument("--items", required=True); f.add_argument("--rows", required=True)
    f.add_argument("--out", required=True); f.add_argument("--workers", type=int, default=3); f.add_argument("--timeout", type=int, default=TIMEOUT)
    f.add_argument("--feedback-out", help="results-style rows (verdict compile, redacted message) for rounds.py generate --results")
    a = ap.parse_args(argv)
    if a.command == "filter":
        out, todo, results = filter_rows(read_jsonl(a.items), read_jsonl(a.rows), lambda p: lower_sandboxed(p, a.py, a.timeout), a.workers,
                                         with_results=True)
        with open(a.out, "w") as fh:
            for r in out:
                fh.write(json.dumps(r) + "\n")
        if a.feedback_out:
            with open(a.feedback_out, "w") as fh:
                for r in feedback_rows(todo, results):
                    fh.write(json.dumps(r) + "\n")
        print(json.dumps(dict(collections.Counter(r["outcome"] for r in out))), flush=True)
        return
    items = validation_set()
    if a.limit:
        items = items[:a.limit]
    print(f"{len(items)} judged kernels: {dict(collections.Counter(c for _, _, c in items))}", flush=True)
    rows, summary = validate(items, lambda p: lower_sandboxed(p, a.py, a.timeout), a.workers)
    pathlib.Path(a.out).write_text(json.dumps({"summary": summary, "rows": [{"id": i, "chip": c, **r} for i, c, r in rows]}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == [CHILD_FLAG]:
        child()
    main()
