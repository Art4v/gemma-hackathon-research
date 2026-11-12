"""The TPU oracle: runs each solution on a TPU v6e and judges it. See oracle/DESCRIPTION.md.

  python oracle/tpu_oracle.py PACK.json OUT.jsonl [--shard I/N] [--limit K] [--seeds 3] [--speed-bar 0.7] [--timeout 480]
                              [--stack S] [--static-only] [--cpu-smoke]

Per solution, in order (the first failure is the verdict):
  1 kernel check  traced at the full input shapes, nothing run: at least one pallas_call, the output depends on one, and no heavy
                  work outside (outside_ops.py's rule). Otherwise not_a_kernel. --static-only stops here (static_ok).
  2 runs          compile, then run on every seed. A failure is classed compile, vmem, mosaic, timeout or runtime, with a short
                  feedback line carrying the chip's trimmed error text; harness_oom is the harness's failure, not the solution's.
  3 correct       on every seed, against a precise answer: the reference on float32 inputs at matmul precision "highest". Correct if
                  np.allclose(rtol=atol=1e-2), or if the error is at most 2 * max(XLA default's own error, 1e-6). Otherwise wrong.
  4 speed         seed 0, median of 20 calls after 3 warm-ups, against jax.jit(reference) at default precision:
                  speed_vs_xla = t_xla / t_kernel. pass if speed_vs_xla >= --speed-bar (0.7, a policy choice), else correct_slow.
Seeds: seed 0 is the task's create_inputs as written; seed s > 0 folds s into every PRNG key create_inputs makes (and every
np.random.default_rng(int) seed), so shapes and dtypes stay the task's and only the draws change.
A task whose inputs are too large for the chip (3 x input bytes > the HBM budget) is too_large_for_chip, not run.
The reference side (answer, XLA error and XLA time) is computed once per task and reused for all of its solutions in the shard.
Resumable: ids already in OUT are skipped. --cpu-smoke runs Pallas in interpret mode on CPU to test the plumbing only: timings mean
nothing. The pure parts (error classes, feedback, the kernel check on a jaxpr, the verdict) import without JAX.
"""
import argparse, ast, contextlib, hashlib, json, os, pathlib, re, signal, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent  # this folder: tpu_check.py and outside_ops.py sit beside it
sys.path[:0] = [str(ROOT)]
from outside_ops import _sub_jaxprs, classify_avals, walk, walk_avals  # noqa: E402

SPEED_BAR, N_SEEDS = 0.7, 3  # a policy choice (Nov 8), not a statistical one: see oracle/DESCRIPTION.md, "The speed bar"
TIMEOUT_S = 480  # per solution, for its three seeds; the reference gets its own budget once per task
FEEDBACK_CHARS = 400
LABELS = {"compile": "compile error", "vmem": "VMEM out of memory", "mosaic": "Mosaic compile error",
          "harness_oom": "harness error (HBM out of memory or the reference failed), not the kernel", "timeout": "timeout",
          "runtime": "runtime error"}
_NOISE = ("Please report a bug", "See https://openxla.org")
ORACLE_FILES = ("tpu_oracle.py", "tpu_check.py", "outside_ops.py")  # the code that decides a verdict: oracle_sha hashes these, in order


# ---- pure: no JAX ----

def classify_error(stage: str, error: str, msg: str) -> str:
    """Error class of a failure at `stage` (ref, static, compile, run, timing). Any reference failure, and an HBM out-of-memory, is
    the harness's (harness_oom); VMEM errors stay the solution's. VMEM is matched before Mosaic because Mosaic messages name
    memref<..., #tpu.memory_space<vmem>> types without being about VMEM size."""
    m = (msg or "").lower()
    if stage == "ref":
        return "harness_oom"
    if error == "Timeout":
        return "timeout"
    if "scoped vmem" in m or "vmem limit" in m or "memory space vmem" in m or ("would exceed memory" in m and "space=vmem" in m):
        return "vmem"
    if error == "MosaicError" or "mosaic failed" in m:
        return "mosaic"
    if ("resource_exhausted" in m and "vmem" not in m) or ("out of memory" in m and "hbm" in m):
        return "harness_oom"
    return "compile" if stage in ("static", "compile") else "runtime"


def trim(msg: str, n: int = FEEDBACK_CHARS) -> str:
    """One line: noise lines dropped, whitespace collapsed; if long, the head (what failed) and the tail (the numbers) are kept."""
    t = " ".join(l.strip() for l in (msg or "").splitlines() if l.strip() and not l.strip().startswith(_NOISE))
    t = re.sub(r"\s+", " ", t)
    return t if len(t) <= n else t[: n // 3] + " ... " + t[-(n - n // 3):]


# HLO attributes (one level of nested braces) that bury the numbers in VMEM messages
_HLO_NOISE = re.compile(r"\b(?:frontend_attributes|metadata)=\{(?:[^{}]|\{[^{}]*\})*\},? ?")
_KEY_NUMBERS = (re.compile(r"Scoped allocation with size \S+ and limit \S+"), re.compile(r"Used \S+ of \S+ vmem"),
                re.compile(r"block shape \((?:[^()]|\([^()]*\))*\), array shape \([^()]*\)"))


def feedback_for(cls: str, msg: str, timeout_s: int | None = None) -> str:
    """One line: the class label, then the chip text trimmed to FEEDBACK_CHARS. For vmem and compile failures the key numbers
    (scoped size and limit, VMEM used of capacity, block shape and array shape) come first, the HLO attributes stripped."""
    if cls == "timeout":
        return f"{LABELS[cls]}: no result within {timeout_s or TIMEOUT_S} s on the TPU"
    key = ""
    if cls in ("vmem", "compile"):
        msg = _HLO_NOISE.sub("", msg or "")
        key = "; ".join(m.group(0) for r in _KEY_NUMBERS for m in [r.search(msg)] if m)
    t = trim(msg, FEEDBACK_CHARS - len(key) - 2) if key else trim(msg)
    t = "; ".join(x for x in (key, t) if x)
    return f"{LABELS[cls]}: {t}" if t else LABELS[cls]


def xla_eq(err_kernel: float | None, err_xla: float | None) -> bool:
    """The kernel's error is at most twice XLA default's own, floored at 1e-6. One-sided: a kernel more accurate than XLA passes."""
    return err_kernel is not None and err_xla is not None and err_kernel <= 2 * max(err_xla, 1e-6)


def _is_var(v) -> bool:
    return not hasattr(v, "val")  # jaxpr Literals carry .val and are unhashable


def _tainted_outputs(jaxpr, tainted_in: list) -> list:
    """For each outvar of `jaxpr`: does it carry data from a pallas_call, given which invars do."""
    t = {v for v, x in zip(jaxpr.invars, tainted_in) if x and _is_var(v)}
    is_t = lambda v: _is_var(v) and v in t
    for e in jaxpr.eqns:
        ins = [is_t(v) for v in e.invars]
        subs = list(_sub_jaxprs(e.params)) if e.primitive.name != "pallas_call" else []
        if len(subs) == 1 and len(subs[0].invars) == len(e.invars) and len(subs[0].outvars) == len(e.outvars):
            t.update(v for v, x in zip(e.outvars, _tainted_outputs(subs[0], ins)) if x)  # jit, closed_call, custom_jvp/vjp, remat
            continue
        # Loops and conds are coarse: every output counts as tainted if any input is or a pallas_call sits inside. A pallas result
        # that dies inside a loop body would pass this rule; the outside-ops rule still has to pass too.
        if e.primitive.name == "pallas_call" or any(ins) or any(walk(s)[1] for s in subs):
            t.update(v for v in e.outvars if _is_var(v))
    return [is_t(v) for v in jaxpr.outvars]


def pallas_reaches_output(jaxpr) -> bool:
    return any(_tainted_outputs(jaxpr, [False] * len(jaxpr.invars)))


def kernel_check(jaxpr) -> dict:
    """Step 1 on a traced jaxpr: pallas_call present, output depends on it, nothing heavy outside (outside_ops.walk_avals and
    classify_avals; outside_metadata_ok lists the primitives let through as metadata)."""
    outside, heavy, n = walk_avals(jaxpr)
    ok, bad, meta = classify_avals(outside, heavy)
    dep = pallas_reaches_output(jaxpr)
    reason = ("no pallas_call" if n == 0 else f"heavy work outside pallas_call: {', '.join(bad)}" if not ok else
              "the output does not depend on any pallas_call" if not dep else "")
    return {"is_kernel": not reason, "reason": reason, "pallas_calls": n, "outside": outside, "outside_disallowed": bad,
            "outside_metadata_ok": meta, "output_depends_on_pallas": dep}


_STATIC = re.compile(r"jax\.jit\(\s*workload\s*,\s*static_argnums\s*=\s*(\([^)]*\)|\[[^\]]*\]|\d+)")


def static_argnums(code: str) -> tuple:
    """The static_argnums of the reference's own `jax.jit(workload, static_argnums=...)`, or () when it has none. Static positions are
    Python values used in shapes; they must stay concrete in every trace and compile, for the reference and the solution alike."""
    m = _STATIC.search(code)
    if not m:
        return ()
    v = ast.literal_eval(m.group(1))
    return (v,) if isinstance(v, int) else tuple(v)


def _g(x) -> str:
    return "n/a" if x is None else f"{x:.3g}"


def final_verdict(per_seed: list, t_kernel: float, t_xla: float, speed_bar: float) -> tuple:
    """(verdict, feedback, speed_vs_xla) once every seed ran. per_seed rows carry seed, ok (correct or xla_eq), err_kernel, err_xla_default."""
    speed = t_xla / t_kernel if t_kernel and t_xla is not None else None
    bad = [s for s in per_seed if not s["ok"]]
    if bad:
        parts = [f"seed {s['seed']}: output shape or structure differs from the reference" if s["err_kernel"] is None else
                 f"seed {s['seed']}: max error {_g(s['err_kernel'])} of max|ref| (XLA default: {_g(s['err_xla_default'])})" for s in bad]
        return "wrong", f"wrong output on {len(bad)} of {len(per_seed)} seeds; " + "; ".join(parts), speed
    if speed is None:
        return "correct_slow", "correct, but the speed could not be measured", None
    if speed >= speed_bar:
        return "pass", f"pass: {speed:.2f}x XLA", speed
    return "correct_slow", f"correct but too slow: {speed:.2f}x XLA, need >= {speed_bar:.2f}x", speed


# ---- JAX: set by init() ----

jax = jnp = np = tc = None


def init(cpu_smoke: bool) -> None:
    """Import JAX and tpu_check. On a TPU, pin JAX to tpu_check.JAX_PIN (install it and restart if it differs)."""
    global jax, jnp, np, tc
    if tc is not None:
        return
    if cpu_smoke:
        os.environ["JAX_PLATFORMS"] = "cpu"
    import tpu_check as _tc
    import jax as _jax, jax.numpy as _jnp, numpy as _np
    if not cpu_smoke and _jax.__version__ != _tc.JAX_PIN:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"jax[tpu]=={_tc.JAX_PIN}"], check=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    jax, jnp, np, tc = _jax, _jnp, _np, _tc


@contextlib.contextmanager
def seeded(seed: int):
    """Seed 0: create_inputs untouched. Seed s > 0: every jax.random.key / PRNGKey it makes is fold_in(key, s), and every
    np.random.default_rng(x) with an integer x is default_rng((x, s)), so shapes, dtypes and distributions stay the task's and only the
    draws change. Inputs built without these RNGs are the same on every seed: seeds_distinct says whether the seeds differ at all."""
    if not seed:
        yield
        return
    r, nr = jax.random, np.random
    key, prng, rng = r.key, r.PRNGKey, nr.default_rng
    r.key = lambda *a, **k: r.fold_in(key(*a, **k), seed)
    r.PRNGKey = lambda *a, **k: r.fold_in(prng(*a, **k), seed)
    # only an int seed is folded; None, a SeedSequence or a Generator passes through as written
    nr.default_rng = lambda x=None, *a, **k: rng((int(x), seed) if isinstance(x, (int, np.integer)) else x, *a, **k)
    try:
        yield
    finally:
        r.key, r.PRNGKey, nr.default_rng = key, prng, rng


def make_inputs(ref, seed: int):
    """The task's own create_inputs(), at its own default dtype, for this seed."""
    with seeded(seed):
        return ref.create_inputs()


def nbytes(tree) -> int:
    return int(sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(tree) if hasattr(x, "dtype")))


def _fingerprint(inputs) -> float:
    return float(sum(jnp.sum(x, dtype=jnp.float32) for x in jax.tree_util.tree_leaves(inputs)
                     if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.number)))


def _same_inputs(a, b) -> bool:
    """Exact, leaf by leaf. _fingerprint is a sum, so two draws of a permutation (always the same sum) look alike to it."""
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(bool(jnp.array_equal(x, y, equal_nan=True)) for x, y in zip(la, lb))


def _host(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


@contextlib.contextmanager
def interpret(on: bool):
    from jax.experimental import pallas as pl
    orig = pl.pallas_call
    if on:
        pl.pallas_call = lambda *a, **k: orig(*a, **{**k, "interpret": True})
    try:
        yield
    finally:
        pl.pallas_call = orig


class Task:
    """One task's reference, loaded once; its chip results computed on first need. A failure is kept and re-raised for every solution."""

    def __init__(self, name: str, code: str):
        self.name, self.code, self.dir = name, code, tempfile.TemporaryDirectory()
        self.loaded = self.chip = None

    def load(self) -> None:
        if self.loaded is None:
            try:
                self.ref = tc.load(self.code, "ref", self.dir.name)
                want, held = static_argnums(self.code), {}

                def build():  # static positions keep their Python values; eval_shape would make them abstract int32[]
                    xs = make_inputs(self.ref, 0)
                    held.update((i, xs[i]) for i in want if i < len(xs))  # a static position may be passed by the harness, not create_inputs
                    return xs
                abstract = jax.eval_shape(build)
                self.static = tuple(sorted(held))
                self.abstract = type(abstract)(held.get(i, x) for i, x in enumerate(abstract)) if held else abstract
                self.info = {"input_dtype": ",".join(sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(self.abstract) if hasattr(x, "dtype")})),
                             "full_input_bytes": nbytes(self.abstract),
                             "shapes": [list(x.shape) for x in jax.tree_util.tree_leaves(self.abstract) if hasattr(x, "shape")],
                             **({"static_argnums": list(self.static)} if self.static else {})}
                self.loaded = True
            except Exception as e:
                self.loaded = e
        if self.loaded is not True:
            raise self.loaded

    def inputs(self, seed: int):
        return make_inputs(self.ref, seed)

    def jit(self, f):
        return jax.jit(f, static_argnums=self.static)

    def dynamic(self, inputs) -> list:
        """The inputs an AOT-compiled workload takes: static positions are baked in at lower() and must be left out."""
        return [x for i, x in enumerate(inputs) if i not in self.static]

    def run(self, seeds: list, timeout_s: int) -> None:
        if self.chip is None:
            signal.alarm(timeout_s)  # the reference's own budget; the solution's starts again below
            try:
                self.chip = self._run(seeds)
            except Exception as e:
                self.chip = e
            signal.alarm(timeout_s)
        if isinstance(self.chip, Exception):
            raise self.chip

    def _key(self, inputs, state: dict):
        """The answer key: a real higher-precision answer, not XLA's own rounding. f32_highest: floating inputs cast to float32 on the
        host, run at matmul precision 'highest'. native_highest: the task's own dtypes at 'highest'. native_default: the key is XLA
        default itself (xla_eq then means matching XLA); only reached when both fail. A level that fails once is not retried on later
        seeds. Returns None for native_default (the caller uses XLA's output)."""
        f32 = lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x
        while state["precision"] != "native_default":
            xs = jax.tree_util.tree_map(f32, inputs) if state["precision"] == "f32_highest" else inputs
            try:
                with jax.default_matmul_precision("highest"):
                    return _host(jax.block_until_ready(self.jit(self.ref.workload)(*xs)))
            except Exception as e:
                state["failed"].append(f"{state['precision']}: {type(e).__name__}")
                state["precision"] = "native_highest" if state["precision"] == "f32_highest" else "native_default"
            finally:
                del xs
        return None

    def _run(self, seeds: list) -> dict:
        f_ref, per_seed, fps, t_xla = self.jit(self.ref.workload), [], [], (None, None, None)
        state = {"precision": "f32_highest", "failed": []}
        for s in seeds:
            inputs = self.inputs(s)
            fps.append(_fingerprint(inputs))
            # the reference and each solution get separate make_inputs calls, so create_inputs must be deterministic per seed (an
            # unseeded default_rng() is fresh entropy each call): built twice, once per task, on the first seed
            if s == seeds[0] and not _same_inputs(inputs, self.inputs(s)):
                raise RuntimeError("create_inputs is not deterministic for a fixed seed (an unseeded RNG?): the reference and the "
                                   "solution would be checked on different inputs")
            want = self._key(inputs, state)
            xla = _host(f_ref(*inputs))
            if want is None:
                want = xla
            per_seed.append({"want": want, "err_xla_default": tc.rel_err(want, xla)})
            if s == seeds[0]:
                t_xla = tc.timed_stats(f_ref, inputs)
            del inputs, xla
        return {"per_seed": per_seed, "t_xla_s": t_xla[0], "t_xla_p10_s": t_xla[1], "t_xla_p90_s": t_xla[2],
                "seeds_distinct": len(set(fps)) == len(fps), "ref_precision": state["precision"], "ref_precision_failed": state["failed"]}


def _provenance() -> dict:
    commit = os.environ.get("ORACLE_COMMIT")
    if not commit:
        p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        commit = p.stdout.strip() if p.returncode == 0 else "unknown"
    # git is absent on a TPU VM and a commit hides local edits: the hash of the files that decide the verdict says exactly what ran
    sha = hashlib.sha256(b"".join((ROOT / p).read_bytes() for p in ORACLE_FILES)).hexdigest()[:12]
    return {"oracle_commit": commit, "oracle_sha": sha}


def evaluate(item: dict, task: Task, a, prov: dict) -> dict:
    seeds = list(range(a.seeds))
    r = {"id": item["id"], "task": item["task"], **{k: item[k] for k in ("model", "arm", "teacher", "ref_sha256") if k in item},
         "stack": a.stack, "device_kind": getattr(jax.devices()[0], "device_kind", "?"), "jax": jax.__version__, **prov,
         "seeds": seeds, "speed_bar": a.speed_bar}
    stage = "ref"
    with tempfile.TemporaryDirectory() as d:
        try:
            task.load()
            r.update(task.info)
            if not a.cpu_smoke and 3 * r["full_input_bytes"] > tc.HBM_BUDGET_BYTES:
                return {**r, "verdict": "too_large_for_chip", "error": "too_large_for_chip",
                        "feedback": "inputs too large for one v6e in this harness (3 x input bytes > HBM budget); not the kernel's fault"}
            stage = "static"
            cand = tc.load(item["code"], "cand", d)
            kc = kernel_check(jax.make_jaxpr(cand.workload, static_argnums=task.static)(*task.abstract).jaxpr)
            reason = kc.pop("reason")
            r.update(kc)
            if not kc["is_kernel"]:
                return {**r, "verdict": "not_a_kernel", "feedback": f"not a Pallas kernel: {reason}. Do the computation inside "
                        "pl.pallas_call; outside it use only layout and dtype ops (reshape, transpose, pad, astype)."}
            if a.static_only:
                return {**r, "verdict": "static_ok", "feedback": ""}
            stage = "ref"
            task.run(seeds, a.timeout)
            c = task.chip
            r.update(seeds_distinct=c["seeds_distinct"], t_xla_s=c["t_xla_s"], t_xla_p10_s=c["t_xla_p10_s"], t_xla_p90_s=c["t_xla_p90_s"],
                     ref_precision=c["ref_precision"], **({"ref_precision_failed": c["ref_precision_failed"]} if c["ref_precision_failed"] else {}))
            stage = "compile"
            compiled = jax.jit(cand.workload, static_argnums=task.static).lower(*task.abstract).compile()
            per_seed = []
            for s, ref_s in zip(seeds, c["per_seed"]):
                stage = "run"
                inputs = task.dynamic(task.inputs(s))
                got = _host(jax.block_until_ready(compiled(*inputs)))
                want, ek = ref_s["want"], tc.rel_err(ref_s["want"], got)
                ok = tc.close(want, got)
                eq = not ok and xla_eq(ek, ref_s["err_xla_default"])
                per_seed.append({"seed": s, "correct": ok, "err_kernel": ek, "err_xla_default": ref_s["err_xla_default"], "xla_eq": eq,
                                 "ok": ok or eq})
                if s == seeds[0]:
                    stage = "timing"
                    r["t_kernel_s"], r["t_kernel_p10_s"], r["t_kernel_p90_s"] = tc.timed_stats(compiled, inputs)
                del inputs, got
            verdict, feedback, speed = final_verdict(per_seed, r["t_kernel_s"], r["t_xla_s"], a.speed_bar)
            return {**r, "per_seed": per_seed, "correct": all(s["ok"] for s in per_seed), "speed_vs_xla": speed,
                    "verdict": verdict, "feedback": feedback}
        except Exception as e:  # compile, VMEM, Mosaic, HBM, lowering, runtime failures and the alarm are results, not crashes
            name = "Timeout" if isinstance(e, tc.Timeout) else type(e).__name__
            cls = classify_error(stage, name, str(e))
            fb = feedback_for(cls, str(e) or name, timeout_s=a.timeout)
            return {**r, "failed_stage": stage, "error": name, "msg": str(e)[-400:], "verdict": cls, "feedback": fb,
                    **({"harness_oom": True} if cls == "harness_oom" else {})}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pack"); ap.add_argument("out")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="correct must hold on seeds 0..N-1 (0 = create_inputs as written)")
    ap.add_argument("--speed-bar", type=float, default=SPEED_BAR, help="pass needs speed_vs_xla = t_xla / t_kernel >= this")
    ap.add_argument("--stack", default="cloud-tpu-vm"); ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    ap.add_argument("--static-only", action="store_true", help="the kernel check only, on CPU: not_a_kernel or static_ok, nothing run")
    ap.add_argument("--cpu-smoke", action="store_true", help="interpret mode on CPU, plumbing only")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    init(cpu_smoke=a.cpu_smoke or a.static_only)
    signal.signal(signal.SIGALRM, tc._alarm)
    refs, items = tc.select(json.loads(pathlib.Path(a.pack).read_text()), a.shard, a.limit)
    out = pathlib.Path(a.out)
    done, prov, dev0 = tc.done_ids(out), _provenance(), jax.devices()[0]
    meta = {"meta": True, "oracle": True, "jax": jax.__version__, "platform": dev0.platform, "device_kind": getattr(dev0, "device_kind", "?"),
            "n_devices": len(jax.devices()), "shard": a.shard, "items": len(items), "seeds": list(range(a.seeds)),
            "speed_bar": a.speed_bar, "timeout_s": a.timeout, **prov,
            **({"static_only": True} if a.static_only else {}), **({"cpu_smoke": True} if a.cpu_smoke else {})}
    print(json.dumps(meta), flush=True)
    task = None
    with out.open("a") as f:
        f.write(json.dumps(meta) + "\n")
        # grouped by task (stable), so each reference is computed once and only one task's outputs are held at a time
        for k, item in sorted(enumerate(items), key=lambda kv: kv[1]["task"]):
            if item["id"] in done:
                continue
            if task is None or task.name != item["task"]:
                task = Task(item["task"], refs[item["task"]])
            signal.alarm(a.timeout)
            with interpret(a.cpu_smoke):  # around the static trace too: jit reuses make_jaxpr's trace (the jaxpr still has pallas_call)
                r = evaluate(item, task, a, prov)
            signal.alarm(0)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(k, r["id"], r["verdict"], r.get("speed_vs_xla"), r["feedback"][:120], flush=True)
            jax.clear_caches()


if __name__ == "__main__":
    main()
