"""The TPU oracle: the single verdict on a (task, kernel), for teacher data and for student evaluation.

Decision (Hruday, Nov 7, issue #1): a kernel counts when it runs on the TPU, matches the reference and matches XLA's speed. The 256-cap
CPU gate, the shrink-8 recheck and the 12 MiB static fit are not verdicts. Everything runs at the task's own create_inputs size on a v6e.

  python oracle/tpu_oracle.py PACK.json OUT.jsonl [--which dev|train] [--shard I/N] [--limit K] [--seeds 3] [--speed-bar 0.8]
                              [--stack S] [--timeout 480] [--static-only] [--reference-only] [--cpu-smoke]
                              [--sensitivity [--sensitivity-seed 1] [--sensitivity-bar X]]

Per kernel, in order (the first failure is the verdict):
  1 is_kernel  (static, traced at the full abstract shapes, nothing executed on a device): at least one pallas_call, the output depends
               on one, and no heavy work outside, per outside_ops.py's oracle rule (walk_avals): layout and dtype ops, and
               primitives whose operands and results are all integer or bool with <= 2**16 elements (index and group metadata);
               a float gather or scatter is heavy. Otherwise not_a_kernel. --static-only stops here (static_ok).
               An item with "human_target": true (a level 2 human kernel, a speed target) only records is_kernel, is_kernel_reason
               and outside; its verdict comes from steps 2-4. Its task's declared adapter (ADAPTERS), if any, maps the level 1
               inputs to its workload's arguments outside the timed call, and the row records it as adapter; an adapter's
               declared seeds limit that item to them (11p: seed 0, gmm's sum(group_sizes) == M), recorded as seeds_used.
  2 runs       lower and compile, then run on every seed. Failures are classed compile, vmem, mosaic, harness_oom, timeout or runtime,
               each with a short feedback line carrying the chip's trimmed error text.
  3 correct    against the reference at matmul precision "highest", tpu_check's loose tolerance, or definition C's xla_eq (one-sided: kernel
               error <= 2*max(XLA-default's own error, 1e-6), SOURCE_OF_TRUTH 5), on EVERY seed. The old two-sided band (within 1% of
               XLA's error) is kept as xla_eq_band for the agreement check against old labels; oracle_sha says which rule a row used.
               Seed 0 is the task's create_inputs as written; seed s > 0 folds s into each PRNG key it makes (and each
               np.random.default_rng(int) seed, as default_rng((x, s))), then applies the task's
               structure transform if structure.py declares one (6p's and 3p's page tables, 11p's group sizes), then redraws the
               constant floating inputs (norm weights and biases, 17p's mask; rows say structure_seed and list float_leaves).
               Every row says int_inputs_vary (whether any integer, bool or static input differs across the seeds run; None when
               there are none), int_leaves (their count) and const_int_leaves (each constant one). Otherwise wrong, with the error
               per seed.
  4 speed      tpu_check.timed_stats (median, p10 and p90 of 20 after 3 warm-ups) on seed 0, against jax.jit(reference) at default
               precision; speed_vs_xla = t_xla / t_kernel, medians. pass iff speed_vs_xla >= --speed-bar, else correct_slow. Rows record
               t_xla_s, t_kernel_s (medians) and t_xla_p10_s, t_xla_p90_s, t_kernel_p10_s, t_kernel_p90_s.
--reference-only (the reference speed table): items need only id and task; per task, the reference alone runs at full size on every seed
and is timed as in step 4, and the row records input and output bytes and fits_hbm; verdict reference_ok, or too_large_for_chip.
--sensitivity (implies --reference-only; Lebron James 6704562752): per task, for each floating input leaf, the reference reruns at
--sensitivity-seed (default 1, the first seed with the structure transforms) with that leaf zeroed, at default precision. One row per
(task, leaf), under the item's id: leaf, dtype, shape, rel_err (tpu_check.rel_err of the zeroed output against the normal one: max abs
difference over max |normal|; None if not finite), err_xla_default at that seed, bar, bar_rule, insensitive (rel_err < bar), verdict
sensitivity. bar = --sensitivity-bar if given, else max(2 * err_xla_default, 0.01): the xla_eq bar floored at tpu_check's loose
tolerance. No kernel runs.
Inputs are built at each task's own create_inputs default (JAXBench p tasks default to bfloat16, so speed is judged at the precision
production runs at); --input-dtype float32 restores tpu_check's old float32 rule.
A task whose own harness has jax.jit(workload, static_argnums=...) (JAXBench 11p) gets the same static positions in every trace,
jit and compile, for the reference and the kernel alike, and its rows record static_argnums.
The reference side (want, XLA error, XLA time per seed) is computed once per task and reused for all of its kernels in the shard.
Pack layout, --shard, --limit and resume (ids already in OUT are skipped) are tpu_check.py's. --cpu-smoke runs Pallas in interpret mode
on CPU to test the plumbing only: no HBM budget, timings mean nothing.
The pure parts (error classes, feedback, the static rule on a jaxpr, the verdict) import without JAX.
"""
import argparse, ast, contextlib, hashlib, inspect, json, math, os, pathlib, re, signal, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent  # this folder: outside_ops.py, tpu_check.py, structure.py and keys.py sit beside it
sys.path[:0] = [str(ROOT)]
from outside_ops import _sub_jaxprs, classify_avals, walk, walk_avals  # noqa: E402
import structure  # noqa: E402  (pure at import: no JAX)
import keys  # noqa: E402  (pure at import: chunked answer keys, e.g. 2p's f32 key that doesn't fit whole)

SPEED_BAR, N_SEEDS = 0.8, 3  # the bar is the humans' to fix; 0.8 until they do
TIMEOUT_S = 480  # per kernel: tpu_check's 240 s covered one seed, this runs three. The reference gets its own budget once per task.
FEEDBACK_CHARS = 400
LABELS = {"compile": "compile error", "vmem": "VMEM out of memory", "mosaic": "Mosaic compile error",
          "harness_oom": "harness error (HBM out of memory or the reference failed), not the kernel", "timeout": "timeout",
          "runtime": "runtime error"}
_NOISE = ("Please report a bug", "See https://openxla.org")


# ---- pure: no JAX ----

def classify_error(stage: str, error: str, msg: str) -> str:
    """Error class of a failure at `stage` (ref, static, compile, run, timing). Same rules as the label rows: any reference failure, and
    an HBM out-of-memory, is the harness's (harness_oom); VMEM errors stay the kernel's. VMEM is matched before Mosaic because Mosaic
    messages name memref<..., #tpu.memory_space<vmem>> types without being about VMEM size."""
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
    """The teacher's line: the class label, then the chip text trimmed to FEEDBACK_CHARS. For vmem and compile failures the key
    numbers (scoped size and limit, VMEM used of capacity, block shape and array shape) come first, the HLO attributes stripped."""
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
    """Definition C (SOURCE_OF_TRUTH 5): the kernel's error is at most twice XLA-default's own, floored at 1e-6. One-sided, so a kernel
    more accurate than XLA passes."""
    return err_kernel is not None and err_xla is not None and err_kernel <= 2 * max(err_xla, 1e-6)


def xla_eq_band(err_kernel: float | None, err_xla: float | None) -> bool:
    """The old two-sided rule, as round2/tpu/v6e_secondary.py's outcome(): the kernel's error equals XLA-default's own to within 1%."""
    return err_kernel is not None and err_xla is not None and abs(err_kernel - err_xla) <= 1e-3 * max(abs(err_xla), 1e-12) * 10


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
        # ponytail: scan/while/cond are coarse: every output counts as tainted if any input is or a pallas_call sits inside. A cheat
        # whose pallas result dies inside a loop body would pass this rule; the outside-ops rule still has to pass too.
        if e.primitive.name == "pallas_call" or any(ins) or any(walk(s)[1] for s in subs):
            t.update(v for v in e.outvars if _is_var(v))
    return [is_t(v) for v in jaxpr.outvars]


def pallas_reaches_output(jaxpr) -> bool:
    return any(_tainted_outputs(jaxpr, [False] * len(jaxpr.invars)))


def kernel_check(jaxpr) -> dict:
    """Step 1 on a traced jaxpr: pallas_call present, output depends on it, nothing heavy outside (outside_ops.walk_avals and
    classify_avals: layout and dtype ops, plus small integer metadata; outside_metadata_ok lists the primitives let through as metadata)."""
    outside, heavy, n = walk_avals(jaxpr)
    ok, bad, meta = classify_avals(outside, heavy)
    dep = pallas_reaches_output(jaxpr)
    reason = ("no pallas_call" if n == 0 else f"heavy work outside pallas_call: {', '.join(bad)}" if not ok else
              "the output does not depend on any pallas_call" if not dep else "")
    return {"is_kernel": not reason, "reason": reason, "pallas_calls": n, "outside": outside, "outside_disallowed": bad,
            "outside_metadata_ok": meta, "output_depends_on_pallas": dep}


def _adapt_paged(xs):
    q, k_pages, v_pages, kv_lens, page_indices, _cu_q_lens = xs
    return (q * q.shape[-1] ** -0.5).astype(q.dtype), k_pages.transpose(2, 0, 1, 3), v_pages.transpose(2, 0, 1, 3), kv_lens, page_indices


# Items with "human_target": true (Lebron James 6704826797) are JAXBench's level 2 human kernels, run as speed targets: is_kernel is
# recorded, not enforced. Where a level 2 workload's arguments differ from level 1's create_inputs, the task's adapter maps the
# level 1 inputs (after the seed and structure transforms) to them, outside the traced, compiled and timed call; the row records it
# as adapter. "seeds", if declared, restricts that human target to those seeds (rows: seeds_used, seed_override_reason); model items
# of the task still run every seed. 1p 2p 3p 4p 7p 8p take level 1's inputs as they are.
ADAPTERS = {
    "6p_Paged_Attention": {"fn": _adapt_paged, "desc": "drop cu_q_lens (arange: one decode token per sequence, never transformed); "
                           "k_pages and v_pages transposed (2, 0, 1, 3) to the kernel's [kv_head, page, slot, dim], as level 2's "
                           "create_inputs does; q pre-scaled by head_dim**-0.5, as level 1 applies it and the upstream kernel expects"},
    "11p_Megablox_GMM": {"fn": lambda xs: tuple(xs[:3]), "desc": "drop max_expert_size (static M // G; gmm tiles by its own TUNED_PARAMS)",
                         "seeds": [0], "seeds_reason": "upstream gmm's contract is sum(group_sizes) == M (rows past the sum are left "
                         "unwritten); only seed 0, JAXBench as written, keeps it, the structure seeds draw sums <= M"},
}


_STATIC = re.compile(r"jax\.jit\(\s*workload\s*,\s*static_argnums\s*=\s*(\([^)]*\)|\[[^\]]*\]|\d+)")


def static_argnums(code: str) -> tuple:
    """The static_argnums of the task's own `jax.jit(workload, static_argnums=...)` (JAXBench 11p: (3,)), or () when it has none."""
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
    """Import JAX and tpu_check (for load, close, rel_err, timed, select, done_ids, the budget). On a TPU, pin JAX as tpu_check does."""
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


def takes_dtype(f) -> bool:
    try:
        ps = inspect.signature(f).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL) for p in ps)


@contextlib.contextmanager
def seeded(seed: int):
    """Seed 0: create_inputs untouched. Seed s > 0: every jax.random.key / PRNGKey it makes is fold_in(key, s), and every
    np.random.default_rng(x) with an integer x is default_rng((x, s)) (51p, 54p, 58p build index structure that way), so shapes, dtypes
    and distributions stay the task's and only the draws change. A create_inputs drawing from another RNG would not vary:
    seeds_distinct and int_inputs_vary say so."""
    if not seed:
        yield
        return
    r, nr = jax.random, np.random
    key, prng, rng = r.key, r.PRNGKey, nr.default_rng
    r.key = lambda *a, **k: r.fold_in(key(*a, **k), seed)
    r.PRNGKey = lambda *a, **k: r.fold_in(prng(*a, **k), seed)
    # ponytail: only an int seed is folded; None, a SeedSequence or a Generator passes through as written
    nr.default_rng = lambda x=None, *a, **k: rng((int(x), seed) if isinstance(x, (int, np.integer)) else x, *a, **k)
    try:
        yield
    finally:
        r.key, r.PRNGKey, nr.default_rng = key, prng, rng


INPUT_DTYPE = None  # None: each task's own create_inputs default; set by --input-dtype float32


def make_inputs(ref, seed: int, dtype=None):
    dtype = dtype if dtype is not None else INPUT_DTYPE
    with seeded(seed):
        return ref.create_inputs(dtype) if dtype is not None and takes_dtype(ref.create_inputs) else ref.create_inputs()


def _is_int(x) -> bool:
    return jnp.issubdtype(x.dtype, jnp.integer) or jnp.issubdtype(x.dtype, jnp.bool_)


def int_leaves(task: str, ref, seed: int) -> dict:
    """{leaf index: value} for the integer and bool array leaves of seed `seed`'s inputs (after its structure transform) and its
    non-array (static Python) leaves. Jitted on the CPU with only those leaves as outputs, so XLA drops the float draws: no chip time,
    and a task's big float inputs are never built. A create_inputs that needs concrete values while traced raises."""
    held = {}

    def f():
        xs = jax.tree_util.tree_leaves(structure.apply(task, make_inputs(ref, seed), seed))
        held.update((i, x) for i, x in enumerate(xs) if not hasattr(x, "dtype"))
        return {i: x for i, x in enumerate(xs) if hasattr(x, "dtype") and _is_int(x)}
    try:
        cpu = jax.devices("cpu")[0]
    except RuntimeError:  # JAX_PLATFORMS without cpu: the default device then
        cpu = jax.devices()[0]
    with jax.default_device(cpu):
        arrs = jax.jit(f)()
    return {**{i: np.asarray(a) for i, a in arrs.items()}, **held}


def _summary(x) -> str:
    if not isinstance(x, np.ndarray):
        return repr(x)
    if x.size and (x == x.flat[0]).all():
        return f"all {x.flat[0]}"
    if x.ndim and np.array_equal(x.ravel(), np.arange(x.size)):
        return "arange"
    return f"fixed, {x.min()}..{x.max()}" if x.size else "empty"


def int_inputs_vary(task: str, ref, seeds: list) -> tuple:
    """(any integer/bool/static leaf differs across `seeds`, None when there are none; [the constant leaves: leaf, dtype, shape,
    summary]; number of such leaves)."""
    per = [int_leaves(task, ref, s) for s in seeds]
    same = lambda a, b: np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
    const = [i for i in sorted(per[0]) if all(same(per[0][i], p[i]) for p in per[1:])]
    desc = [{"leaf": i, "dtype": str(getattr(per[0][i], "dtype", type(per[0][i]).__name__)), "shape": list(np.shape(per[0][i])),
             "summary": _summary(per[0][i])} for i in const]
    return (len(const) < len(per[0]) if per[0] else None), desc, len(per[0])


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
    """One task's reference, loaded once; its chip results computed on first need. A failure is kept and re-raised for every kernel."""

    def __init__(self, name: str, code: str):
        self.name, self.code, self.dir = name, code, tempfile.TemporaryDirectory()
        self.loaded = self.chip = self.int_vary = self.floats = None

    def load(self) -> None:
        if self.loaded is None:
            try:
                self.ref = tc.load(self.code, "ref", self.dir.name)
                want, held = static_argnums(self.code), {}

                def build():  # static positions keep their Python values; eval_shape would make them abstract int32[]
                    xs = make_inputs(self.ref, 0)
                    held.update((i, xs[i]) for i in want if i < len(xs))  # 55p's static 7 is passed by its benchmark, not create_inputs
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
        return structure.apply(self.name, make_inputs(self.ref, seed), seed, self.float_plan()["keep"] if seed else ())

    def float_plan(self) -> dict:
        """The constant floating leaves seeds >= 1 redraw (structure.constant_floats on seed 1's inputs), decided once per task so every
        seed and kernel uses the same ones; a leaf whose draw makes the reference's seed-1 output non-finite is skipped and recorded."""
        if self.floats is None:
            xs = structure.apply(self.name, make_inputs(self.ref, 1), 1)
            cand, f = structure.constant_floats(xs), self.jit(self.ref.workload)
            finite = lambda ls: all(bool(jnp.all(jnp.isfinite(y))) for y in jax.tree_util.tree_leaves(f(*structure.apply(self.name, xs, 1, ls)))
                                    if jnp.issubdtype(y.dtype, jnp.inexact))
            # ponytail: finiteness is checked on seed 1 only (one reference run); a draw non-finite only on a later seed would show as
            # a reference with NaNs there. Per-leaf retry only when the joint draw fails.
            keep = cand if not cand or finite(cand) else [i for i in cand if finite([i])]
            ls = jax.tree_util.tree_leaves(xs)
            info = lambda i: {"leaf": i, "dtype": str(ls[i].dtype), "shape": list(ls[i].shape), "value": float(ls[i].reshape(-1)[0])}
            self.floats = {"keep": keep, "varied": [info(i) for i in keep], "skipped": [info(i) for i in cand if i not in keep]}
            del xs, ls
        return self.floats

    def float_fields(self) -> dict:
        """structure_seed, the combined description and float_leaves (float_leaves_skipped) once seeds >= 1 have been built; else {}."""
        f = self.floats
        if not f or not (f["varied"] or f["skipped"]):
            return {}
        desc = "; ".join(d for d in (structure.describe(self.name), structure.describe_floats(self.name, f["varied"])) if d)
        return {**({"structure_seed": True, "structure": desc} if f["varied"] else {}), "float_leaves": f["varied"],
                **({"float_leaves_skipped": f["skipped"]} if f["skipped"] else {})}

    def structure_fields(self, seeds: list) -> dict:
        """structure_seed, the transform, and int_inputs_vary for these seeds (computed once per task). A failure to compute
        int_inputs_vary is recorded as None with its reason; it never decides a verdict."""
        if self.int_vary is None:
            try:
                vary, const, n = int_inputs_vary(self.name, self.ref, seeds)
                self.int_vary = {"int_inputs_vary": vary, "int_leaves": n, "const_int_leaves": const}
            except Exception as e:
                self.int_vary = {"int_inputs_vary": None, "int_inputs_vary_error": f"{type(e).__name__}: {str(e)[:200]}"}
        desc = structure.describe(self.name)
        return {**({"structure_seed": True, "structure": desc} if desc and any(seeds) else {}), **self.int_vary}

    def jit(self, f):
        return jax.jit(f, static_argnums=self.static)

    def dynamic(self, inputs) -> list:
        """The inputs an AOT-compiled workload takes: static positions are baked in at lower() and must be left out."""
        return [x for i, x in enumerate(inputs) if i not in self.static]

    def run(self, seeds: list, timeout_s: int) -> None:
        if self.chip is None:
            signal.alarm(timeout_s)  # the reference's own budget; the kernel's starts again below
            try:
                self.chip = self._run(seeds)
            except Exception as e:
                self.chip = e
            signal.alarm(timeout_s)
        if isinstance(self.chip, Exception):
            raise self.chip

    def _key(self, inputs, state: dict):
        """The answer key: a real higher-precision answer, not XLA's own rounding (clanker 6704124776, Lebron James 6704045873).
        f32_highest: floating inputs cast to float32 on the host, run at matmul precision 'highest' (Mosaic never involved for a plain
        reference). native_highest: the task's own dtypes at 'highest'. native_default: the key is XLA default itself (xla_eq then
        means bit-for-bit XLA); only reached when both fail, e.g. a level 2 stand-in containing Pallas. A level that fails once is
        not retried on later seeds. Returns None for native_default (the caller uses XLA's output)."""
        f32 = lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x
        while state["precision"] != "native_default":
            xs = jax.tree_util.tree_map(f32, inputs) if state["precision"] == "f32_highest" else inputs
            try:
                # a key too big to compute whole at f32 (2p: 34 GB of scores) is the same maths in chunks, from oracle/keys.py
                chunked = state["precision"] == "f32_highest" and self.name in keys.KEYS
                fn = keys.KEYS[self.name] if chunked else self.ref.workload
                with jax.default_matmul_precision("highest"):
                    want = _host(jax.block_until_ready(self.jit(fn)(*xs)))
                state["chunked_key"] = chunked
                return want
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
            # the reference and each kernel get separate make_inputs calls, so create_inputs must be deterministic per seed (an
            # unseeded default_rng() is fresh entropy each call): built twice, once per task, on the first seed (Lebron James 6711257577)
            if s == seeds[0] and not _same_inputs(inputs, self.inputs(s)):
                raise RuntimeError("create_inputs is not deterministic for a fixed seed (an unseeded RNG?): the reference and the "
                                   "kernel would be checked on different inputs")
            want = self._key(inputs, state)
            xla = _host(f_ref(*inputs))
            if want is None:
                want = xla
            per_seed.append({"want": want, "err_xla_default": tc.rel_err(want, xla), "ref_default_tight": tc.close(want, xla, True)})
            if s == seeds[0]:
                t_xla = tc.timed_stats(f_ref, inputs)
            del inputs, xla
        precision = state["precision"]
        return {"per_seed": per_seed, "t_xla_s": t_xla[0], "t_xla_p10_s": t_xla[1], "t_xla_p90_s": t_xla[2], "seeds_distinct": len(set(fps)) == len(fps), "ref_precision": precision,
                "ref_precision_failed": state["failed"], **({"ref_key_chunked": True} if state.get("chunked_key") else {})}


def _provenance() -> dict:
    commit = os.environ.get("ORACLE_COMMIT")
    if not commit:
        p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        commit = p.stdout.strip() if p.returncode == 0 else "unknown"
    # git is absent on a TPU VM and a commit hides local edits: the hash of the four files that decide the verdict says exactly what ran
    src = [pathlib.Path(__file__), pathlib.Path(tc.__file__), ROOT / "outside_ops.py", ROOT / "structure.py", ROOT / "keys.py"]
    sha = hashlib.sha256(b"".join(p.read_bytes() for p in src if p.exists())).hexdigest()[:12]
    return {"oracle_commit": commit, "oracle_sha": sha}


def reference_row(r: dict, task: Task, seeds: list, a) -> dict:
    """--reference-only: the task's reference alone, on every seed, timed. Failures raise to evaluate(), at stage "ref"."""
    fits = 3 * r["full_input_bytes"] <= tc.HBM_BUDGET_BYTES
    r = {**r, "output_bytes": nbytes(task.jit(task.ref.workload).eval_shape(*task.abstract)), "fits_hbm": fits,
         "hbm_budget_bytes": tc.HBM_BUDGET_BYTES}
    if not fits and not a.cpu_smoke:
        return {**r, "verdict": "too_large_for_chip", "feedback": "3 x input bytes > HBM budget: not run"}
    task.run(seeds, a.timeout)
    c = task.chip
    return {**r, **task.float_fields(), "seeds_distinct": c["seeds_distinct"], "t_xla_s": c["t_xla_s"],
            "t_xla_p10_s": c["t_xla_p10_s"], "t_xla_p90_s": c["t_xla_p90_s"], "ref_precision": c["ref_precision"],
            **({"ref_precision_failed": c["ref_precision_failed"]} if c["ref_precision_failed"] else {}),
            **({"ref_key_chunked": True} if c.get("ref_key_chunked") else {}), "verdict": "reference_ok", "feedback": "",
            "per_seed": [{"seed": s, "err_xla_default": p["err_xla_default"], "ref_default_tight": p["ref_default_tight"]}
                         for s, p in zip(seeds, c["per_seed"])]}


SENSITIVITY_FLOOR = 1e-2  # tpu_check's loose tolerance


def sensitivity_rows(r: dict, task: Task, a) -> list:
    """--sensitivity: per floating input leaf, rel_err of the reference with that leaf zeroed against the normal output, at seed
    a.sensitivity_seed, default precision both. Failures raise to evaluate(), at stage "ref", as one error row."""
    if 3 * r["full_input_bytes"] > tc.HBM_BUDGET_BYTES and not a.cpu_smoke:
        return {**r, "verdict": "too_large_for_chip", "feedback": "3 x input bytes > HBM budget: not run"}
    s, f = a.sensitivity_seed, task.jit(task.ref.workload)
    inputs = task.inputs(s)
    normal = _host(f(*inputs))
    want = task._key(inputs, {"precision": "f32_highest", "failed": []})
    err = tc.rel_err(want, normal) if want is not None else 0.0
    bar, rule = (a.sensitivity_bar, "fixed") if a.sensitivity_bar is not None else (max(2 * err, SENSITIVITY_FLOOR),
                                                                                    f"max(2 * err_xla_default, {SENSITIVITY_FLOOR:g})")
    leaves, tree = jax.tree_util.tree_flatten(inputs)
    base = {**r, **task.float_fields(), "sensitivity_seed": s, "err_xla_default": err, "bar": bar, "bar_rule": rule}
    rows = []
    for i, x in enumerate(leaves):
        if not (hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)):
            continue
        zeroed = jax.tree_util.tree_unflatten(tree, [jnp.zeros_like(y) if j == i else y for j, y in enumerate(leaves)])
        e = tc.rel_err(normal, _host(f(*zeroed)))
        e = e if e is not None and math.isfinite(e) else None
        rows.append({**base, "leaf": i, "dtype": str(x.dtype), "shape": list(x.shape), "rel_err": e,
                     "insensitive": e is not None and e < bar, "verdict": "sensitivity", "feedback": ""})
        del zeroed
    return rows or [{**base, "leaf": None, "rel_err": None, "insensitive": False, "verdict": "sensitivity",
                     "feedback": "no floating input"}]  # one row anyway, so resume and collect see the task done


def evaluate(item: dict, task: Task, a, prov: dict) -> dict | list:
    seeds = list(range(a.seeds))
    human = item.get("human_target") is True
    adapter = ADAPTERS.get(item["task"]) if human else None
    used = [s for s in seeds if s in adapter["seeds"]] if adapter and "seeds" in adapter else seeds
    r = {"id": item["id"], "task": item["task"], **{k: item[k] for k in ("model", "arm", "teacher", "ref_sha256") if k in item},
         **({"human_target": True} if human else {}), **({"adapter": adapter["desc"]} if adapter else {}),
         **({"seeds_used": used, "seed_override_reason": adapter["seeds_reason"]} if used != seeds else {}),
         "stack": a.stack, "device_kind": getattr(jax.devices()[0], "device_kind", "?"), "jax": jax.__version__, **prov,
         "seeds": seeds, "speed_bar": a.speed_bar}
    stage = "ref"
    with tempfile.TemporaryDirectory() as d:
        try:
            task.load()
            r.update(task.info)
            r.update(task.structure_fields(seeds))
            if a.sensitivity:
                return sensitivity_rows(r, task, a)
            if a.reference_only:
                return reference_row(r, task, seeds, a)
            if not a.cpu_smoke and 3 * r["full_input_bytes"] > tc.HBM_BUDGET_BYTES:
                return {**r, "verdict": "too_large_for_chip", "error": "too_large_for_chip",
                        "feedback": "inputs too large for one v6e in this harness (3 x input bytes > HBM budget); not the kernel's fault"}
            stage = "static"
            cand = tc.load(item["code"], "cand", d)
            if adapter:  # the adapter drops 11p's static position, so the adapted call has none
                c_abstract, c_static = jax.eval_shape(adapter["fn"], tuple(task.abstract)), ()
                c_inputs = lambda s: adapter["fn"](tuple(task.inputs(s)))
            else:
                c_abstract, c_static, c_inputs = task.abstract, task.static, lambda s: task.dynamic(task.inputs(s))
            kc = kernel_check(jax.make_jaxpr(cand.workload, static_argnums=c_static)(*c_abstract).jaxpr)
            reason = kc.pop("reason")
            r.update(kc, **({"is_kernel_reason": reason} if human else {}))
            if not kc["is_kernel"] and not human:
                return {**r, "verdict": "not_a_kernel", "feedback": f"not a Pallas kernel: {reason}. Do the computation inside "
                        "pl.pallas_call; outside it use only layout and dtype ops (reshape, transpose, pad, astype)."}
            if a.static_only:
                return {**r, "verdict": "static_ok", "feedback": ""}
            stage = "ref"
            task.run(seeds, a.timeout)
            r.update(task.float_fields())
            r.update(seeds_distinct=task.chip["seeds_distinct"], t_xla_s=task.chip["t_xla_s"], t_xla_p10_s=task.chip["t_xla_p10_s"],
                     t_xla_p90_s=task.chip["t_xla_p90_s"], ref_precision=task.chip["ref_precision"],
                     **({"ref_precision_failed": task.chip["ref_precision_failed"]} if task.chip["ref_precision_failed"] else {}),
                     **({"ref_key_chunked": True} if task.chip.get("ref_key_chunked") else {}))
            stage = "compile"
            compiled = jax.jit(cand.workload, static_argnums=c_static).lower(*c_abstract).compile()
            per_seed = []
            for s, ref_s in ((s, x) for s, x in zip(seeds, task.chip["per_seed"]) if s in used):
                stage = "run"
                inputs = c_inputs(s)
                got = _host(jax.block_until_ready(compiled(*inputs)))
                want, ek = ref_s["want"], tc.rel_err(ref_s["want"], got)
                c = tc.close(want, got, False)
                row = {"seed": s, "correct": c, "tight_correct": tc.close(want, got, True), "err_kernel": ek,
                       "err_xla_default": ref_s["err_xla_default"], "ref_default_tight": ref_s["ref_default_tight"],
                       "xla_eq": not c and xla_eq(ek, ref_s["err_xla_default"])}
                per_seed.append({**row, "ok": c or row["xla_eq"]})
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
    ap.add_argument("--which", default="dev"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="correct must hold on seeds 0..N-1 (0 = create_inputs as written)")
    ap.add_argument("--speed-bar", type=float, default=SPEED_BAR, help="pass needs speed_vs_xla = t_xla / t_kernel >= this")
    ap.add_argument("--stack", default="colab-tpu"); ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    ap.add_argument("--input-dtype", choices=["native", "float32"], default="native",
                    help="native: each task's create_inputs default (bfloat16 for JAXBench p tasks); float32: the old tpu_check rule")
    ap.add_argument("--static-only", action="store_true", help="step 1 only, on CPU: not_a_kernel or static_ok, nothing run")
    ap.add_argument("--reference-only", action="store_true", help="time each task's reference alone at full size; no kernel")
    ap.add_argument("--cpu-smoke", action="store_true", help="interpret mode on CPU, plumbing only")
    ap.add_argument("--sensitivity", action="store_true", help="reference only: rerun it with each floating input zeroed, one row each")
    ap.add_argument("--sensitivity-seed", type=int, default=1, help="the one seed the sensitivity check runs at")
    ap.add_argument("--sensitivity-bar", type=float, default=None, help="insensitive iff rel_err < this; default max(2 * err_xla_default, 0.01)")
    a = ap.parse_args(argv)
    a.reference_only = a.reference_only or a.sensitivity
    return a


def main(argv=None) -> None:
    a = parse_args(argv)
    init(cpu_smoke=a.cpu_smoke or a.static_only)
    global INPUT_DTYPE
    INPUT_DTYPE = jnp.float32 if a.input_dtype == "float32" else None
    signal.signal(signal.SIGALRM, tc._alarm)
    refs, items = tc.select(json.loads(pathlib.Path(a.pack).read_text()), a.which, a.shard, a.limit)
    out = pathlib.Path(a.out)
    done, prov, dev0 = tc.done_ids(out), _provenance(), jax.devices()[0]
    meta = {"meta": True, "oracle": True, "jax": jax.__version__, "platform": dev0.platform, "device_kind": getattr(dev0, "device_kind", "?"),
            "n_devices": len(jax.devices()), "which": a.which, "shard": a.shard, "items": len(items), "seeds": list(range(a.seeds)),
            "speed_bar": a.speed_bar, "timeout_s": a.timeout, "input_dtype_rule": a.input_dtype, **prov,
            **({"static_only": True} if a.static_only else {}), **({"reference_only": True} if a.reference_only else {}),
            **({"cpu_smoke": True} if a.cpu_smoke else {}),
            **({"sensitivity": True, "sensitivity_seed": a.sensitivity_seed, "sensitivity_bar": a.sensitivity_bar} if a.sensitivity else {})}
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
            for r in (r if isinstance(r, list) else [r]):  # --sensitivity: one row per floating input
                f.write(json.dumps(r) + "\n"); f.flush()
                print(k, r["id"], r["verdict"], r.get("speed_vs_xla"), r["feedback"][:120], flush=True)
            jax.clear_caches()


if __name__ == "__main__":
    main()
