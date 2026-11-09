"""Dataset v3: the CPU check for a (reference, kernel) pair. Nothing here has been on a chip, and rows that rely on it say so (meta.v3.verification = cpu).

Per item, in one isolated child per batch, using the oracle's own Task, kernel check and closeness test:
  ref        loads, inputs fit the HBM budget, create_inputs is deterministic and differs across the 3 seeds, the outputs are finite
  signature  the op chain and multiset of the reference (signature.py's walk), so filter_contamination does not start a process per problem
  static     the oracle's step 1 at full shapes: a pallas_call, the output depends on it, no heavy work outside
  lowering   jax.jit(workload).lower(lowering_platforms=("tpu",)) really lowers the kernel to Mosaic (no VMEM accounting: that needs the chip)
  numerics   interpret mode, 3 seeds, at full shapes, against the oracle's answer key (float32 inputs, matmul precision "highest"), tc.close

Results are appended to OUT as they finish, so a stopped run resumes. One child at a time; each is capped at --max-rss-gb.
"""
from __future__ import annotations

import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ORACLE, Isolated, read_jsonl  # noqa: E402
from signature import _SIGPURE  # noqa: E402
from v3_common import ORACLE_FILES  # noqa: E402

CHILD = r'''
import json, os, signal, sys, tempfile, threading, time
sys.path.insert(0, ".")
import tpu_oracle as o
o.init(cpu_smoke=True)
import sigpure
jax, jnp, np, tc = o.jax, o.jnp, o.np, o.tc
signal.signal(signal.SIGALRM, tc._alarm)
BUDGET = int(sys.argv[1])
# Some lowering rules ask the device for its TPU spec, and the host device here is a CPU: answer with one v6e core, the chip the oracle judges on.
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V6E, 1)

def check(it):
    r = {"id": it["id"], "stage": "ref_load"}
    t = o.Task("task", it["ref_code"])
    t.load()
    r.update(input_bytes=t.info["full_input_bytes"], input_shapes=t.info["shapes"], input_dtypes=t.info["input_dtype"])
    if 3 * t.info["full_input_bytes"] > tc.HBM_BUDGET_BYTES:
        return {**r, "ok": False, "stage": "fits_hbm", "error": "too large"}
    r["stage"] = "ref_inputs"
    x0 = t.inputs(0)
    if not o._same_inputs(x0, t.inputs(0)):
        return {**r, "ok": False, "error": "create_inputs not deterministic"}
    r["stage"] = "signature"
    pr = jax.make_jaxpr(t.ref.workload, static_argnums=t.static)(*t.abstract).jaxpr
    ops = sigpure.walk_ops(pr)
    outs = jax.eval_shape(t.jit(t.ref.workload), *t.abstract)
    leaves = [x for x in jax.tree_util.tree_leaves(t.abstract) if hasattr(x, "shape")]
    r["signature"] = {"op_chain": sigpure.collapse(ops), "op_multiset": sigpure.multiset(ops), "input_shapes": [list(x.shape) for x in leaves],
                      "input_dtypes": [str(x.dtype) for x in leaves], "output_shapes": [list(x.shape) for x in jax.tree_util.tree_leaves(outs)],
                      "input_bytes": t.info["full_input_bytes"]}
    with tempfile.TemporaryDirectory() as d:
        r["stage"] = "static"
        cand = tc.load(it["code"], "cand", d)
        kc = o.kernel_check(jax.make_jaxpr(cand.workload, static_argnums=t.static)(*t.abstract).jaxpr)
        r["static_verdict"] = "static_ok" if kc["is_kernel"] else "not_a_kernel"
        r["static_reason"] = kc["reason"]
        r["pallas_calls"] = kc["pallas_calls"]
        if not kc["is_kernel"]:
            return {**r, "ok": False, "error": kc["reason"]}
        r["stage"] = "lowering"
        jax.jit(cand.workload, static_argnums=t.static).trace(*t.abstract).lower(lowering_platforms=("tpu",))
        r["lowered"] = True
        jax.clear_caches()
        r["stage"] = "numerics"
        fps, per, state = [], [], {"precision": "f32_highest", "failed": []}
        with o.interpret(True):
            f = jax.jit(cand.workload, static_argnums=t.static)
            for s in range(3):
                x = t.inputs(s)
                fps.append(o._fingerprint(x))
                want = t._key(x, state)
                if want is None:
                    want = o._host(t.jit(t.ref.workload)(*x))
                got = o._host(jax.block_until_ready(f(*x)))
                for leaf in jax.tree_util.tree_leaves(want):
                    if np.issubdtype(np.asarray(leaf).dtype, np.floating) and not np.isfinite(np.asarray(leaf, np.float32)).all():
                        return {**r, "ok": False, "error": "reference output not finite"}
                per.append({"seed": s, "correct": bool(tc.close(want, got)), "err": tc.rel_err(want, got)})
                del x, want, got
        r.update(per_seed=per, ref_precision=state["precision"], seeds_distinct=len(set(fps)) == len(fps))
        if not r["seeds_distinct"]:
            return {**r, "ok": False, "error": "inputs do not differ across seeds"}
        wrong = [p for p in per if not p["correct"]]
        if wrong:
            return {**r, "ok": False, "error": f"wrong on {len(wrong)} of 3 seeds, max err {max((p['err'] or 0) for p in wrong):.3g}"}
    return {**r, "ok": True, "stage": "done"}

def out_of_time():  # the alarm can't interrupt native code, so a watchdog ends the process; the parent charges the item it was on and carries on after it
    print(f"[budget: over {BUDGET}s]", file=sys.stderr, flush=True)
    os._exit(9)

for it in json.load(open("items.json")):
    signal.alarm(BUDGET)
    guard = threading.Timer(BUDGET, out_of_time)
    guard.daemon = True
    guard.start()
    t0 = time.time()
    r = {"id": it["id"], "stage": "?"}
    try:
        r = check(it)
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        r = {"id": it["id"], "ok": False, "stage": "timeout" if isinstance(e, tc.Timeout) else "error", "error": f"{type(e).__name__}: {str(e)[:400]}"}
    signal.alarm(0)
    guard.cancel()
    r["seconds"] = round(time.time() - t0, 2)
    print("@@" + json.dumps(r), flush=True)
    jax.clear_caches()
'''


def run_batch(batch: list, python: str | None, budget: int, max_rss_gb: float) -> tuple:
    files = {p: (ORACLE / p).read_text() for p in ORACLE_FILES}
    files.update({"child.py": CHILD, "sigpure.py": _SIGPURE, "items.json": json.dumps(batch)})
    r = Isolated(python, timeout=60 + budget * len(batch), max_rss_gb=max_rss_gb).run(["child.py", str(budget)], files=files, extra_env={"ORACLE_COMMIT": "v3-verify"})
    rows = [json.loads(l[2:]) for l in r.stdout.splitlines() if l.startswith("@@")]
    return rows, (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1][-200:]


def verify(items: list, out: pathlib.Path, python: str | None, *, batch: int = 25, budget: int = 30, max_rss_gb: float = 6.0, log=print) -> dict:
    """{id: result} for every item, resuming from `out`. A child that dies charges the crash to the item it was on and continues after it."""
    done = {r["id"]: r for r in read_jsonl(out)}
    todo = [i for i in items if i["id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"verify: {len(items)} items, {len(todo)} to do")
    k = 0
    with out.open("a") as f:
        while k < len(todo):
            chunk = todo[k:k + batch]
            rows, tail = run_batch(chunk, python, budget, max_rss_gb)
            if len(rows) < len(chunk):  # the child died on the item after the last one it reported
                dead = chunk[len(rows)]
                rows.append({"id": dead["id"], "ok": False, "stage": "crash", "error": tail})
            for r in rows:
                f.write(json.dumps(r) + "\n")
                done[r["id"]] = r
            f.flush()
            k += len(rows)
            log(f"  verified {k} of {len(todo)}; ok so far {sum(1 for r in done.values() if r.get('ok'))}")
    return done
