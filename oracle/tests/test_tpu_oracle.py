"""oracle/tpu_oracle.py: the TPU-only verdict on a (task, kernel). The pure parts (error classes, feedback, the speed bar, the static
is-a-kernel rule on a jaxpr) are tested without JAX on fake jaxprs and injected timings. The JAX tests run the oracle on CPU in Pallas
interpret mode through a driver that replaces the timer, on small hand-written kernels only."""
import importlib.util, json, os, pathlib, re, subprocess, sys, tempfile, textwrap, unittest

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
PY = os.environ.get("SERA_JAX_PY") or (sys.executable if importlib.util.find_spec("jax") else "")

import tpu_oracle as o  # noqa: E402  (pure at import: no JAX)


class P:
    def __init__(self, name): self.name = name


class V:  # a jaxpr Var: hashable by identity
    pass


class Lit:  # a jaxpr Literal: unhashable, has .val
    __hash__ = None
    def __init__(self, val): self.val = val


class Eqn:
    def __init__(self, name, invars, outvars, params=None):
        self.primitive, self.invars, self.outvars, self.params = P(name), invars, outvars, params or {}


class Jaxpr:
    def __init__(self, invars, eqns, outvars): self.invars, self.eqns, self.outvars = invars, eqns, outvars


class Closed:
    def __init__(self, jaxpr): self.jaxpr = jaxpr


def pallas(i, o_):
    return Eqn("pallas_call", [i], [o_], {"jaxpr": Jaxpr([V()], [Eqn("dot_general", [], [])], [])})


class OutputDependsOnPallas(unittest.TestCase):
    def test_pallas_output_that_reaches_the_result_counts(self):
        x, p, y = V(), V(), V()
        j = Jaxpr([x], [pallas(x, p), Eqn("reshape", [p], [y])], [y])
        self.assertTrue(o.pallas_reaches_output(j))

    def test_a_pallas_call_whose_output_is_dropped_does_not_count(self):
        x, p, y = V(), V(), V()
        j = Jaxpr([x], [pallas(x, p), Eqn("transpose", [x], [y])], [y])
        self.assertFalse(o.pallas_reaches_output(j))

    def test_an_input_returned_as_is_does_not_count(self):
        x, p = V(), V()
        self.assertFalse(o.pallas_reaches_output(Jaxpr([x], [pallas(x, p)], [x])))

    def test_literals_are_skipped(self):
        x, p, y = V(), V(), V()
        j = Jaxpr([x], [pallas(x, p), Eqn("add", [p, Lit(1.0)], [y])], [y])
        self.assertTrue(o.pallas_reaches_output(j))

    def test_follows_data_through_a_jit_sub_jaxpr(self):
        a, b = V(), V()
        inner_live = Jaxpr([a], [pallas(a, b)], [b])
        c, d = V(), V()
        inner_dead = Jaxpr([c], [pallas(c, V())], [c])
        x, y, z = V(), V(), V()
        self.assertTrue(o.pallas_reaches_output(Jaxpr([x], [Eqn("jit", [x], [y], {"jaxpr": Closed(inner_live)})], [y])))
        self.assertFalse(o.pallas_reaches_output(Jaxpr([x], [Eqn("jit", [x], [z], {"jaxpr": Closed(inner_dead)})], [z])))

    def test_taint_flows_into_a_sub_jaxpr_through_its_inputs(self):
        a, b = V(), V()
        inner = Jaxpr([a], [Eqn("reshape", [a], [b])], [b])
        x, p, y = V(), V(), V()
        j = Jaxpr([x], [pallas(x, p), Eqn("jit", [p], [y], {"jaxpr": Closed(inner)})], [y])
        self.assertTrue(o.pallas_reaches_output(j))


class KernelCheck(unittest.TestCase):
    def test_no_pallas_call_is_not_a_kernel(self):
        x, y = V(), V()
        r = o.kernel_check(Jaxpr([x], [Eqn("dot_general", [x, x], [y])], [y]))
        self.assertEqual((r["is_kernel"], r["pallas_calls"]), (False, 0))
        self.assertIn("no pallas_call", r["reason"])

    def test_heavy_work_outside_is_not_a_kernel_and_names_the_ops(self):
        x, p, y = V(), V(), V()
        r = o.kernel_check(Jaxpr([x], [pallas(x, p), Eqn("mul", [p, p], [y])], [y]))
        self.assertFalse(r["is_kernel"])
        self.assertEqual(r["outside_disallowed"], ["mul"])
        self.assertIn("mul", r["reason"])

    def test_output_not_from_pallas_is_not_a_kernel(self):
        x, p = V(), V()
        r = o.kernel_check(Jaxpr([x], [pallas(x, p)], [x]))
        self.assertFalse(r["is_kernel"])
        self.assertFalse(r["output_depends_on_pallas"])
        self.assertIn("does not depend", r["reason"])

    def test_honest_kernel_passes(self):
        x, p, y = V(), V(), V()
        r = o.kernel_check(Jaxpr([x], [pallas(x, p), Eqn("reshape", [p], [y])], [y]))
        self.assertEqual((r["is_kernel"], r["reason"], r["pallas_calls"], r["outside_disallowed"]), (True, "", 1, []))


# Real chip messages, from the committed v6e label rows.
MOSAIC = ("INTERNAL: Mosaic failed to compile TPU kernel: infer-vector-layout: unsupported shape cast\n\nThe MLIR operation involved:\n"
          "  %41 = \"tpu.reshape\"(%40) : (vector<8x64xi1>) -> vector<8x64x1xi1>\n\nPlease report a bug at: https://github.com/jax-ml/jax/issues/new/choose\n")
MOSAIC_MEMREF = ("Mosaic failed to compile TPU kernel: failed to legalize (memref<1x1x2048x1024xf32, #tpu.tiled<(8,128),[2048,2048,8,1]>, "
                 "#tpu.memory_space<vmem>>, index) -> vector<1x1x512x256xf32>\n\nPlease report a bug at: https://github.com/jax-ml/jax/issues/new/choose\n")
VMEM = ("RESOURCE_EXHAUSTED: " + "f32[8,1024,4096]{2,1,0} " * 40 + "metadata={op_name=\"jit(workload)/pallas_call\" stack_frame_id=3}. "
        "Scoped allocation with size 37.00M and limit 32.00M exceeded scoped vmem limit by 5.00M. It should not be possible to run out of "
        "scoped vmem - \nSee https://openxla.org/xla/errors/error_1001 for more details.")
VMEM_WINDOW = ("size=268435456) would exceed memory (size=134217728) :: #allocation4 [shape = 'u8[268435456]{0}', space=vmem, size = 0x10000000, "
               "scoped, tag = 'input window allocation for operator input 1. The window shape is f32[8192,8192]'] :: workload.1")
HBM = "RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total memory required for HLO temporaries (512.19G) exceeds available HBM (31.25G)."


class ErrorClasses(unittest.TestCase):
    def test_classes(self):
        cases = [("compile", "MosaicError", MOSAIC, "mosaic"), ("compile", "MosaicError", MOSAIC_MEMREF, "mosaic"),
                 ("compile", "JaxRuntimeError", VMEM, "vmem"), ("run", "JaxRuntimeError", VMEM_WINDOW, "vmem"),
                 ("run", "JaxRuntimeError", HBM, "harness_oom"), ("run", "Timeout", "", "timeout"),
                 ("compile", "TypeError", "dot_general requires contracting dimensions to have the same shape", "compile"),
                 ("static", "ValueError", "no kernel here", "compile"), ("run", "XlaRuntimeError", "INTERNAL: core halted", "runtime"),
                 ("timing", "XlaRuntimeError", "INTERNAL: core halted", "runtime"), ("ref", "TypeError", "anything", "harness_oom")]
        for stage, err, msg, want in cases:
            self.assertEqual(o.classify_error(stage, err, msg), want, (stage, err, msg[:60]))

    def test_feedback_is_short_names_the_class_and_keeps_the_chip_text(self):
        for cls, msg, key in [("mosaic", MOSAIC, "unsupported shape cast"), ("vmem", VMEM, "exceeded scoped vmem limit by 5.00M"),
                              ("harness_oom", HBM, "exceeds available HBM"), ("compile", "no kernel here", "no kernel here")]:
            fb = o.feedback_for(cls, msg)
            self.assertLessEqual(len(fb), 600, fb)
            self.assertIn(key, fb)
            self.assertTrue(fb.lower().startswith(o.LABELS[cls].lower()), fb)
            self.assertNotIn("Please report a bug", fb)
            self.assertNotIn("\n", fb)

    def test_timeout_feedback_says_how_long(self):
        self.assertIn("480", o.feedback_for("timeout", "", timeout_s=480))


# Real chip messages from the Nov 7 runs. VMEM_USED was classed compile before "memory space vmem" was matched.
VMEM_SCOPED = ("VMEM out of memory: RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom: Ran out of memory in memory space vmem while "
               "allocating on stack for %workload.2 = (f32[8,1024,4096]{2,1,0}, f32[8,4096]{1,0}) custom-call(" + "bf16[8,1024,4096]{2,1,0} %p, " * 30
               + "), custom_call_target=\"tpu_custom_call\", frontend_attributes={kernel_metadata={}}, metadata={op_name=\"jit(workload)/pallas_call\" "
               "stack_frame_id=9}. Scoped allocation with size 37.50M and limit 32.00M exceeded scoped vmem limit by 5.50M. It should not be "
               "possible to run out of scoped vmem - please file a bug against XLA.\nSee https://openxla.org/xla/errors/error_1001 for more details.")
VMEM_USED = ("RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space vmem. Used 238.51M of 127.94M vmem. "
             "Exceeded vmem capacity by 110.57M.\n\nTotal vmem usage: " + "allocation of size 16.00M for f32[8,128,4096] in operand 3; " * 25)
BLOCK = ("The Pallas TPU lowering currently requires that the last two dimensions of your block shape are divisible by 8 and 128 "
         "respectively, or be equal to the respective dimensions of the overall array. Block spec for args[0] in pallas_call "
         "_kernel at /tmp/cand.py:12 has block shape (Blocked(block_size=324), Blocked(block_size=640)), array shape (9720, 3200), and "
         "index_map { lambda ; a:i32[] b:i32[]. let  in (a, b) }, in memory space <MemorySpace.VMEM: 'vmem'>." + " padding" * 60)


class FeedbackNumbersFirst(unittest.TestCase):
    """For vmem and compile failures the teacher sees the numbers before any trimmed part, with the HLO attribute noise gone."""

    def test_used_of_vmem_message_is_vmem_not_compile(self):
        self.assertEqual(o.classify_error("compile", "JaxRuntimeError", VMEM_USED), "vmem")
        self.assertEqual(o.classify_error("compile", "ValueError", BLOCK), "compile")  # its memory space repr is not "memory space vmem"
        self.assertEqual(o.classify_error("compile", "MosaicError", MOSAIC_MEMREF), "mosaic")  # VMEM still before Mosaic, Mosaic still Mosaic

    def test_the_numbers_come_first_and_the_length_is_bounded(self):
        for cls, msg, key in [("vmem", VMEM_SCOPED, "Scoped allocation with size 37.50M and limit 32.00M"),
                              ("vmem", VMEM_USED, "Used 238.51M of 127.94M vmem"),
                              ("compile", BLOCK, "block shape (Blocked(block_size=324), Blocked(block_size=640)), array shape (9720, 3200)")]:
            fb = o.feedback_for(cls, msg)
            self.assertTrue(fb.startswith(o.LABELS[cls] + ": " + key), fb)
            self.assertIn(" ... ", fb)  # the rest is still trimmed, after the numbers
            self.assertLessEqual(len(fb), len(o.LABELS[cls]) + 2 + len(o.trim("x" * 10 ** 4)), fb)  # the old bound
            self.assertNotIn("\n", fb)

    def test_hlo_attribute_noise_is_stripped(self):
        fb = o.feedback_for("vmem", VMEM_SCOPED)
        self.assertNotIn("frontend_attributes", fb)
        self.assertNotIn("metadata={", fb)
        self.assertNotIn("stack_frame_id", fb)

    def test_a_message_without_numbers_is_trimmed_as_before(self):
        self.assertEqual(o.feedback_for("compile", "no kernel here"), "compile error: no kernel here")
        self.assertEqual(o.feedback_for("mosaic", MOSAIC), "Mosaic compile error: " + o.trim(MOSAIC))


class AV:  # a jaxpr Var with an aval
    def __init__(self, dtype, shape=()): self.aval = type("Aval", (), {"dtype": dtype, "shape": shape})()


class KernelCheckMetadata(unittest.TestCase):
    def test_small_int_metadata_outside_is_allowed_and_listed(self):
        g, off, x, p = AV("int32", (8,)), AV("int32", (8,)), AV("float32", (64, 128)), AV("float32", (64, 128))
        r = o.kernel_check(Jaxpr([g, x], [Eqn("cumsum", [g], [off]), pallas(x, p)], [p]))
        self.assertEqual((r["is_kernel"], r["outside"], r["outside_disallowed"], r["outside_metadata_ok"]),
                         (True, {"cumsum": 1}, [], ["cumsum"]), r)

    def test_a_float_gather_outside_is_not_a_kernel(self):
        pages, idx, kv, p = AV("float32", (16, 8, 128)), AV("int32", (4, 1)), AV("float32", (4, 8, 128)), AV("float32", (4, 8, 128))
        r = o.kernel_check(Jaxpr([pages, idx], [Eqn("gather", [pages, idx], [kv]), pallas(kv, p)], [p]))
        self.assertEqual((r["is_kernel"], r["outside_disallowed"], r["outside_metadata_ok"]), (False, ["gather"], []), r)
        self.assertIn("gather", r["reason"])


TIMED_BASE = "76aca6d"  # origin/jaxbench-study when timed_stats was added; the frozen study imports timed()


class TimedUnchanged(unittest.TestCase):
    def test_timed_is_byte_identical_to_the_frozen_version(self):
        import ast
        old = subprocess.run(["git", "-C", str(HERE), "show", f"{TIMED_BASE}:./paper/tpu/tpu_check.py"], capture_output=True, text=True)
        if old.returncode:
            self.skipTest(f"{TIMED_BASE} not in this clone")
        seg = lambda src: next(ast.get_source_segment(src, n) for n in ast.parse(src).body if getattr(n, "name", "") == "timed")
        self.assertEqual(seg((HERE / "tpu_check.py").read_text()), seg(old.stdout))


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class TimedStats(unittest.TestCase):
    def test_median_p10_p90_are_ordered_over_the_same_calls_as_timed(self):
        script = textwrap.dedent('''
            import json, sys, time
            sys.path.insert(0, sys.argv[1])
            import tpu_check as tc
            n = [0]
            def f(x):
                n[0] += 1
                time.sleep(0.001 * (n[0] % 5))
                return x
            med, p10, p90 = tc.timed_stats(f, (1.0,))
            print("JSON" + json.dumps({"stats": [med, p10, p90], "calls": n[0], "want": tc.WARMUP + tc.REPS}))
        ''')
        p = subprocess.run([PY, "-c", script, str(HERE)], capture_output=True, text=True, timeout=300,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
        v = json.loads(next(l[4:] for l in p.stdout.splitlines() if l.startswith("JSON")))
        med, p10, p90 = v["stats"]
        self.assertTrue(0 < p10 <= med <= p90, v)
        self.assertLess(p10, p90, v)  # sleeps of 0..4 ms: the spread is real
        self.assertEqual(v["calls"], v["want"])


class XlaEq(unittest.TestCase):
    """Definition C (SOURCE_OF_TRUTH 5, clanker 6704178439 on Lebron James 6704176225): one-sided, kernel error <= 2*max(XLA's, 1e-6)."""

    def test_kernel_more_accurate_than_xla_passes(self):
        self.assertTrue(o.xla_eq(0.03, 0.05))
        rows = seed_rows(True, True, True)
        ek, ex = 0.03, 0.05
        rows[1].update(correct=False, err_kernel=ek, err_xla_default=ex, xla_eq=o.xla_eq(ek, ex), ok=o.xla_eq(ek, ex))
        self.assertEqual(o.final_verdict(rows, t_kernel=1.0, t_xla=1.0, speed_bar=0.8)[0], "pass")

    def test_three_times_xla_error_fails(self):
        self.assertFalse(o.xla_eq(0.15, 0.05))

    def test_exactly_twice_xla_error_is_the_inclusive_boundary(self):
        self.assertTrue(o.xla_eq(0.1, 0.05))
        self.assertFalse(o.xla_eq(0.1000001, 0.05))

    def test_floor_of_1e_6_when_xla_is_exact(self):
        self.assertTrue(o.xla_eq(1.5e-6, 0.0))
        self.assertFalse(o.xla_eq(3e-6, 0.0))

    def test_missing_errors_are_not_equal(self):
        self.assertFalse(o.xla_eq(None, 0.01))
        self.assertFalse(o.xla_eq(0.5, None))


class XlaEqBand(unittest.TestCase):
    """The old two-sided rule (round2/tpu/v6e_secondary.py), kept for the agreement check against old labels."""

    def test_within_one_percent_of_xla_error(self):
        self.assertTrue(o.xla_eq_band(0.0024312385, 0.0024312440))
        self.assertTrue(o.xla_eq_band(0.01009, 0.01))
        self.assertFalse(o.xla_eq_band(0.0102, 0.01))
        self.assertFalse(o.xla_eq_band(0.005, 0.01))
        self.assertFalse(o.xla_eq_band(None, 0.01))
        self.assertFalse(o.xla_eq_band(0.5, None))


def seed_rows(*oks):
    return [{"seed": s, "correct": ok, "xla_eq": False, "ok": ok, "err_kernel": 0.0 if ok else 0.25, "err_xla_default": 0.001}
            for s, ok in enumerate(oks)]


class Verdict(unittest.TestCase):
    """Speed bar with injected timings: speed_vs_xla = t_xla / t_kernel, pass iff correct on every seed and speed >= bar."""

    def test_fast_correct_kernel_passes(self):
        v, fb, speed = o.final_verdict(seed_rows(True, True, True), t_kernel=1.0, t_xla=1.25, speed_bar=0.8)
        self.assertEqual((v, speed), ("pass", 1.25))

    def test_bar_is_inclusive(self):
        self.assertEqual(o.final_verdict(seed_rows(True, True, True), t_kernel=1.25, t_xla=1.0, speed_bar=0.8)[0], "pass")

    def test_slow_correct_kernel_is_correct_slow_with_the_numbers(self):
        v, fb, speed = o.final_verdict(seed_rows(True, True, True), t_kernel=2.0, t_xla=1.0, speed_bar=0.8)
        self.assertEqual((v, speed), ("correct_slow", 0.5))
        self.assertEqual(fb, "correct but too slow: 0.50x XLA, need >= 0.80x")

    def test_bar_is_a_parameter(self):
        self.assertEqual(o.final_verdict(seed_rows(True, True, True), t_kernel=1.0, t_xla=1.25, speed_bar=1.5)[0], "correct_slow")
        self.assertEqual(o.SPEED_BAR, 0.7)

    def test_default_bar_is_0_7_inclusive(self):
        """The policy bar (oracle/DESCRIPTION.md): exactly 0.7x XLA passes, just under it is correct_slow."""
        rows = seed_rows(True, True, True)
        self.assertEqual(o.final_verdict(rows, t_kernel=1.0, t_xla=0.7, speed_bar=o.SPEED_BAR)[0], "pass")
        v, fb, _ = o.final_verdict(rows, t_kernel=1.0, t_xla=0.69, speed_bar=o.SPEED_BAR)
        self.assertEqual((v, fb), ("correct_slow", "correct but too slow: 0.69x XLA, need >= 0.70x"))

    def test_wrong_on_any_seed_is_wrong_and_names_the_seed_and_error(self):
        v, fb, speed = o.final_verdict(seed_rows(True, False, True), t_kernel=1.0, t_xla=5.0, speed_bar=0.8)
        self.assertEqual((v, speed), ("wrong", 5.0))
        self.assertIn("seed 1", fb)
        self.assertIn("0.25", fb)
        self.assertNotIn("seed 0", fb)

    def test_xla_eq_seed_counts_as_correct(self):
        rows = seed_rows(True, True, True)
        rows[2].update(correct=False, xla_eq=True, ok=True)
        self.assertEqual(o.final_verdict(rows, t_kernel=1.0, t_xla=1.0, speed_bar=0.8)[0], "pass")


# ---- JAX: the oracle end to end on CPU (interpret mode), timer replaced ----

REF = '''import jax, jax.numpy as jnp
def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    return jax.random.uniform(k1, (16, 128), dtype), jax.random.normal(k2, (128,), dtype)
def workload(x, b):
    return x * 2.0 + b
'''
REF_NOARG = REF.replace("def create_inputs(dtype=jnp.float32):", "def create_inputs():").replace(", dtype)", ", jnp.float32)")
HDR = "import jax, jax.numpy as jnp\nfrom jax.experimental import pallas as pl\n"
CALL = '''
def _call(k, x, b):
    return pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), grid=(2,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0)), pl.BlockSpec((1, 128), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)))(x, b.reshape(1, 128))
'''
KERNELS = {
    "plain_jnp": HDR + "def workload(x, b):\n    return x * 2.0 + b\n",
    "good": HDR + CALL + "def _k(x_ref, b_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2.0 + b_ref[...]\n"
                         "def workload(x, b):\n    return _call(_k, x, b)\n",
    # memorizes seed 0: exact while b[0] is seed 0's value, off by 1.0 everywhere otherwise
    "seed0_only": HDR + CALL + "C = {c!r}\ndef _k(x_ref, b_ref, o_ref):\n    b = b_ref[...]\n"
                               "    o_ref[...] = x_ref[...] * 2.0 + b + jnp.where(jnp.abs(b[0, 0] - C) < 1e-7, 0.0, 1.0)\n"
                               "def workload(x, b):\n    return _call(_k, x, b)\n",
    "dropped": HDR + CALL + "def _k(x_ref, b_ref, o_ref):\n    o_ref[...] = x_ref[...]\n"
                            "def workload(x, b):\n    _call(_k, x, b)\n    return x\n",
    "raises": HDR + "def workload(x, b):\n    raise ValueError('no kernel here')\n",
}

DRIVER = '''import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import tpu_oracle as o
o.init(cpu_smoke=True)
import jax
b0 = jax.random.normal(jax.random.split(jax.random.key(0))[1], (128,), jax.numpy.float32)
pack = json.loads(pathlib.Path(sys.argv[2]).read_text())
for it in pack["dev"]:
    it["code"] = it["code"].replace("{c!r}", repr(float(b0[0])))
pathlib.Path(sys.argv[2]).write_text(json.dumps(pack))
o.tc.timed = lambda f, inputs: 1.0  # the timer is mocked: kernel and XLA both 1.0 s, so speed_vs_xla is 1.0
o.tc.timed_stats = lambda f, inputs: (1.0, 0.9, 1.2)  # (median, p10, p90)
o.main(sys.argv[2:])
'''


def run_oracle(items, *flags, refs=None, out_rows=None, oracle_dir=HERE, env=None):
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "pack.json").write_text(json.dumps({"dev_refs": refs or {"t": REF}, "train_refs": {}, "dev": items, "train": []}))
        (d / "driver.py").write_text(DRIVER)
        if out_rows:
            (d / "out.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
        p = subprocess.run([PY, str(d / "driver.py"), str(oracle_dir), str(d / "pack.json"), str(d / "out.jsonl"), "--cpu-smoke", *flags],
                           capture_output=True, text=True, timeout=600, env={**os.environ, "JAX_PLATFORMS": "cpu", **(env or {})})
        assert (d / "out.jsonl").exists(), p.stderr[-1500:]
        lines = (d / "out.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in lines]
    return [r for r in rows if r.get("meta")], [r for r in rows if not r.get("meta")], p


def items(*names, task="t"):
    return [{"id": n, "task": task, "code": KERNELS[n], "teacher": "x"} for n in names]


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metas, rows, cls.proc = run_oracle(items("plain_jnp", "good", "seed0_only", "dropped", "raises"))
        cls.rows = {r["id"]: r for r in rows}

    def test_every_kernel_gets_one_row_with_the_provenance_fields(self):
        self.assertEqual(set(self.rows), {"plain_jnp", "good", "seed0_only", "dropped", "raises"}, self.proc.stderr[-800:])
        for r in self.rows.values():
            for k in ("id", "task", "verdict", "feedback", "seeds", "device_kind", "jax", "oracle_commit", "speed_bar", "teacher"):
                self.assertIn(k, r, (k, r))
            self.assertEqual(r["seeds"], [0, 1, 2])
            self.assertEqual(r["speed_bar"], 0.7)
        m = self.metas[0]
        self.assertTrue(m["meta"] and m["oracle"])
        self.assertEqual((m["seeds"], m["speed_bar"]), ([0, 1, 2], 0.7))

    def test_plain_jnp_is_not_a_kernel_and_never_runs(self):
        r = self.rows["plain_jnp"]
        self.assertEqual((r["verdict"], r["is_kernel"], r["pallas_calls"]), ("not_a_kernel", False, 0), r)
        self.assertIn("no pallas_call", r["feedback"])
        self.assertNotIn("per_seed", r)

    def test_output_that_ignores_the_pallas_call_is_not_a_kernel(self):
        r = self.rows["dropped"]
        self.assertEqual((r["verdict"], r["output_depends_on_pallas"]), ("not_a_kernel", False), r)

    def test_correct_kernel_is_correct_on_every_seed_and_passes_at_the_mocked_speed(self):
        r = self.rows["good"]
        self.assertEqual(r["verdict"], "pass", r)
        self.assertTrue(r["correct"] and r["is_kernel"] and r["seeds_distinct"])
        self.assertEqual([s["seed"] for s in r["per_seed"]], [0, 1, 2])
        self.assertTrue(all(s["correct"] for s in r["per_seed"]))
        self.assertEqual((r["t_kernel_s"], r["t_xla_s"], r["speed_vs_xla"]), (1.0, 1.0, 1.0))
        self.assertEqual((r["t_xla_p10_s"], r["t_xla_p90_s"], r["t_kernel_p10_s"], r["t_kernel_p90_s"]), (0.9, 1.2, 0.9, 1.2))
        self.assertEqual(r["shapes"], [[16, 128], [128]])

    def test_kernel_right_on_one_seed_only_is_wrong_with_per_seed_errors(self):
        r = self.rows["seed0_only"]
        self.assertEqual(r["verdict"], "wrong", r)
        ps = {s["seed"]: s for s in r["per_seed"]}
        self.assertTrue(ps[0]["correct"])
        self.assertFalse(ps[1]["correct"] or ps[2]["correct"])
        self.assertGreater(ps[1]["err_kernel"], 0.05)
        self.assertIn("seed 1", r["feedback"])

    def test_raised_exception_is_a_classified_row_with_feedback(self):
        r = self.rows["raises"]
        self.assertEqual((r["verdict"], r["failed_stage"], r["error"]), ("compile", "static", "ValueError"), r)
        self.assertIn("no kernel here", r["feedback"])

    def test_speed_bar_flag(self):
        _, rows, p = run_oracle(items("good"), "--speed-bar", "1.5")
        self.assertEqual(rows[0]["verdict"], "correct_slow", p.stderr[-800:])
        self.assertEqual(rows[0]["feedback"], "correct but too slow: 1.00x XLA, need >= 1.50x")

    def test_resume_skips_ids_already_in_out(self):
        prior = [{"meta": True}, {"id": "good", "task": "t", "verdict": "pass", "sentinel": 1}]
        metas, rows, p = run_oracle(items("good", "plain_jnp"), out_rows=prior)
        self.assertEqual([r["id"] for r in rows], ["good", "plain_jnp"], p.stderr[-800:])
        self.assertEqual(rows[0].get("sentinel"), 1)
        self.assertEqual(len(metas), 2)

    def test_create_inputs_without_a_dtype_argument_is_handled(self):
        _, rows, p = run_oracle(items("good"), refs={"t": REF_NOARG})
        self.assertEqual(rows[0]["verdict"], "pass", (rows, p.stderr[-800:]))
        self.assertTrue(rows[0]["seeds_distinct"])

    def test_static_only_runs_nothing(self):
        _, rows, p = run_oracle(items("good", "plain_jnp"), "--static-only")
        v = {r["id"]: r["verdict"] for r in rows}
        self.assertEqual(v, {"good": "static_ok", "plain_jnp": "not_a_kernel"}, p.stderr[-800:])
        self.assertNotIn("per_seed", rows[0])


ORACLE_BASE_BLOB = "92d23f8c9b62c08a77c146eed517e92447b549af"  # oracle/tpu_oracle.py at 1761d73 (native dtype), just before --reference-only


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class ByteCompat(unittest.TestCase):
    def test_without_reference_only_the_rows_are_byte_identical_to_1761d73(self):
        """The 1761d73 oracle (by blob hash, so any branch holding it works) and this one, on the same pack, write the same bytes once
        each one's oracle_sha (a hash of its own source) is masked."""
        old = subprocess.run(["git", "-C", str(HERE), "cat-file", "blob", ORACLE_BASE_BLOB], capture_output=True)
        if old.returncode:
            self.skipTest("blob of the 1761d73 oracle not in this clone")
        out = {}
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "oracle").mkdir(); (root / "paper/tpu").mkdir(parents=True)
            (root / "oracle/tpu_oracle.py").write_bytes(old.stdout)
            (root / "outside_ops.py").write_bytes((HERE / "outside_ops.py").read_bytes())
            (root / "paper/tpu/tpu_check.py").write_bytes((HERE / "tpu_check.py").read_bytes())
            for name, od in (("old", root / "oracle"), ("new", HERE)):
                metas, rows, p = run_oracle(items(*KERNELS), oracle_dir=od, env={"ORACLE_COMMIT": "fixed"})
                self.assertEqual(len(rows), len(KERNELS), p.stderr[-800:])
                sha = rows[0]["oracle_sha"]
                # the intended additions since the baseline: ref_precision (fallback when 'highest' won't compile) and
                # int_inputs_vary, int_leaves and const_int_leaves (structure seeds; these toy tasks have no structure transform and
                # no constant float input, so nothing else changes); the timing percentiles and outside_metadata_ok (batch 3)
                extra = ("ref_precision", "int_inputs_vary", "int_leaves", "const_int_leaves", "outside_metadata_ok",
                         "t_xla_p10_s", "t_xla_p90_s", "t_kernel_p10_s", "t_kernel_p90_s")
                rows = [{k: v for k, v in r.items() if k not in extra}
                        for r in rows]
                out[name] = "\n".join(json.dumps(r) for r in metas + rows).replace(sha, "SHA")
        self.assertEqual(out["old"], out["new"])


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class Estimate(unittest.TestCase):
    def test_bytes_fit_and_time_for_a_small_task(self):
        with tempfile.TemporaryDirectory() as d:
            assets = pathlib.Path(d, "assets.json")
            assets.write_text(json.dumps({"dev": {"t": REF, "t_noarg": REF_NOARG}}))
            p = subprocess.run([PY, str(HERE / "estimate.py"), "--assets", str(assets), "--json"], capture_output=True, text=True,
                               timeout=300, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        self.assertEqual(p.returncode, 0, p.stderr[-800:])
        rows = {r["task"]: r for r in map(json.loads, p.stdout.splitlines()) if "task" in r}
        r = rows["t"]
        self.assertEqual((r["input_bytes"], r["output_bytes"], r["fits_hbm"]), (16 * 128 * 4 + 128 * 4, 16 * 128 * 4, True))
        self.assertGreater(r["ref_only_s"], 0)
        self.assertGreater(r["kernel_eval_s"], 0)
        self.assertIn("t_noarg", rows)


if __name__ == "__main__":
    unittest.main()


class NativeDtype(unittest.TestCase):
    """Each task runs at its own create_inputs default (JAXBench p tasks default to bfloat16); --input-dtype float32 restores the old rule."""
    PROBE = textwrap.dedent('''
        import sys, types
        sys.path.insert(0, sys.argv[1])
        import tpu_oracle as o
        o.init(cpu_smoke=True)
        import jax.numpy as jnp
        m = types.ModuleType("ref")
        exec("import jax, jax.numpy as jnp\\ndef create_inputs(dtype=jnp.bfloat16):\\n"
             "    return (jax.random.normal(jax.random.key(0), (8, 128), dtype),)\\n", m.__dict__)
        print(o.make_inputs(m, 0)[0].dtype, o.make_inputs(m, 1, dtype=jnp.float32)[0].dtype)
    ''')

    def test_cli_default_is_native(self):
        self.assertEqual(o.parse_args(["p.json", "o.jsonl"]).input_dtype, "native")
        self.assertEqual(o.parse_args(["p.json", "o.jsonl", "--input-dtype", "float32"]).input_dtype, "float32")

    @unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
    def test_native_default_keeps_bfloat16_and_override_gives_float32(self):
        p = subprocess.run([PY, "-c", self.PROBE, str(HERE)], capture_output=True, text=True, timeout=300)
        self.assertEqual(p.stdout.split(), ["bfloat16", "float32"], p.stderr[-2000:])


BOOM = '''class _Boom:
    def __enter__(self): raise RuntimeError("contract_precision<fp32> is not supported here")
    def __exit__(self, *a): return False
o.jax.default_matmul_precision = lambda *a, **k: _Boom()
'''
REF_DATACLASS = "from __future__ import annotations\n" + REF.replace("import jax, jax.numpy as jnp\n", "import dataclasses, jax, jax.numpy as jnp\n@dataclasses.dataclass\nclass Cfg:\n    n: int = 128\n", 1)


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class ReferenceRobustness(unittest.TestCase):
    """Two TPU-session failures: a reference whose 'highest'-precision run won't compile (Mosaic rejects fp32 contract precision on
    bf16) falls back to default precision and says so; a reference that defines a dataclass loads (its module is registered)."""

    def test_highest_precision_failure_falls_back_to_default_and_is_recorded(self):
        global DRIVER
        old = DRIVER
        DRIVER = DRIVER.replace("o.main(sys.argv[2:])", BOOM + "o.main(sys.argv[2:])")
        try:
            _, rows, p = run_oracle(items("good"))
        finally:
            DRIVER = old
        self.assertEqual(rows[0]["verdict"], "pass", p.stderr[-1500:])
        self.assertEqual(rows[0]["ref_precision"], "native_default", rows[0])

    def test_normal_reference_records_highest(self):
        _, rows, p = run_oracle(items("good"))
        self.assertEqual(rows[0]["ref_precision"], "f32_highest", p.stderr[-1500:])

    def test_reference_with_a_dataclass_loads(self):
        probe = textwrap.dedent('''
            import sys, tempfile
            sys.path.insert(0, sys.argv[1])
            import tpu_oracle as o
            o.init(cpu_smoke=True)
            code = "from __future__ import annotations\\nimport dataclasses\\n@dataclasses.dataclass\\nclass Cfg:\\n    n: int = 1\\n"
            o.tc.load(code, "ref_dc_probe", tempfile.mkdtemp())
            print("LOADED")
        ''')
        p = subprocess.run([PY, "-c", probe, str(HERE)], capture_output=True, text=True, timeout=300,
                           env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(HERE)})
        self.assertIn("LOADED", p.stdout, p.stderr[-1500:])


# 11p_Megablox_GMM's harness jits workload with static_argnums=(3,): input 3 is a Python int used in a shape, so it must stay concrete.
REF_STATIC = '''import jax, jax.numpy as jnp
def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    n = 16
    return jax.random.uniform(k1, (n * 128,), dtype), jax.random.normal(k2, (128,), dtype), n
def workload(x, b, n):
    return x.reshape(n, 128) * 2.0 + b + jnp.zeros((n, 128), x.dtype)
def benchmark():
    fn = jax.jit(workload, static_argnums=(2,))
    return fn(*create_inputs())
'''
GOOD_STATIC = HDR + CALL + ("def _k(x_ref, b_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2.0 + b_ref[...]\n"
                            "def workload(x, b, n):\n    return _call(_k, x.reshape(n, 128), b)\n")
# 55p's pattern: the static position is an argument the harness passes itself, beyond what create_inputs returns
REF_STATIC_BEYOND = REF.replace("def workload(x, b):", "def workload(x, b, k=None):") + \
    "def benchmark():\n    return jax.jit(workload, static_argnums=(2,))(*create_inputs(), 3)\n"


class StaticArgnumsParse(unittest.TestCase):
    def test_reads_the_harness_jit_of_workload(self):
        self.assertEqual(o.static_argnums("fn = jax.jit(workload, static_argnums=(3,))"), (3,))
        self.assertEqual(o.static_argnums("fn = jax.jit(workload, static_argnums=2)"), (2,))
        self.assertEqual(o.static_argnums("fn = jax.jit( workload,\n  static_argnums=[1, 4])"), (1, 4))
        self.assertEqual(o.static_argnums("fn = jax.jit(workload)\ng = jax.jit(other, static_argnums=(1,))"), ())
        self.assertEqual(o.static_argnums(REF), ())


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class StaticArgs(unittest.TestCase):
    def test_kernel_for_a_reference_with_a_static_arg_passes_and_the_row_records_it(self):
        _, rows, p = run_oracle([{"id": "good_static", "task": "t", "code": GOOD_STATIC, "teacher": "x"}], refs={"t": REF_STATIC})
        r = rows[0]
        self.assertEqual(r["verdict"], "pass", (r, p.stderr[-1500:]))
        self.assertEqual(r["static_argnums"], [2])
        self.assertTrue(all(s["correct"] for s in r["per_seed"]) and r["seeds_distinct"])
        self.assertEqual(r["shapes"], [[2048], [128]])

    def test_reference_only_on_a_reference_with_a_static_arg(self):
        _, rows, p = run_oracle([{"id": "t", "task": "t", "code": ""}], "--reference-only", refs={"t": REF_STATIC})
        r = rows[0]
        self.assertEqual(r["verdict"], "reference_ok", (r, p.stderr[-1500:]))
        self.assertEqual((r["static_argnums"], r["output_bytes"]), ([2], 16 * 128 * 4))

    def test_tasks_without_static_args_record_nothing(self):
        for ref in (REF, REF_STATIC_BEYOND):
            _, rows, p = run_oracle(items("good"), refs={"t": ref})
            self.assertEqual(rows[0]["verdict"], "pass", (rows, p.stderr[-1500:]))
            self.assertNotIn("static_argnums", rows[0])


REF_BF16 = REF.replace("def create_inputs(dtype=jnp.float32):", "def create_inputs(dtype=jnp.bfloat16):")
KERNEL_BF16_GOOD = KERNELS["good"]


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class AnswerKey(unittest.TestCase):
    """clanker 6704124776 / Lebron James 6704045873: the key is a real higher-precision answer. A bf16 plain reference is run on
    float32-cast inputs at 'highest' (f32_highest), so XLA's own bf16 output has a non-zero error against it and xla_eq keeps its meaning."""

    def test_bf16_plain_reference_gets_an_f32_highest_key(self):
        _, rows, p = run_oracle(items("good"), refs={"t": REF_BF16})
        r = rows[0]
        self.assertEqual(r["ref_precision"], "f32_highest", (r, p.stderr[-1500:]))
        self.assertGreater(max(s["err_xla_default"] for s in r["per_seed"]), 0.0, r)


# ---- structure seeds (Lebron James 6704387016, clanker 6704395827): integer and structural inputs that fold_in never touches ----

LEAK = HERE.parent / "benchmarks/jaxbench/level1"


def _shrunk(task: str, cfg: dict) -> str:
    """The vendored JAXBench reference, byte for byte except CONFIG's numbers, so the real create_inputs and workload run at toy size."""
    text = (LEAK / task / "baseline.py").read_text()
    for k, v in cfg.items():
        text, n = re.subn(rf"'{k}': \d+,", f"'{k}': {v},", text)
        assert n == 1, (task, k)
    return text


PAGED = "6p_Paged_Attention"
GMM = "11p_Megablox_GMM"
MLA = "3p_MLA_Attention"
REF_PAGED = _shrunk(PAGED, {"num_seqs": 2, "max_seq_len": 32, "num_query_heads": 2, "num_kv_heads": 1, "head_dim": 128,
                            "page_size": 8, "pages_per_seq": 4})
REF_MLA = _shrunk(MLA, {"batch_size": 2, "kv_len_val": 512, "page_size": 256})
REF_GMM = _shrunk(GMM, {"num_experts": 4, "num_experts_per_tok": 1, "emb_dim": 128, "moe_mlp_dim": 128, "seq_len": 32})

PAGED_KERNEL = HDR + '''
def _gather(p, i):
    return GATHER
def _k(q_ref, k_ref, v_ref, l_ref, p_ref, o_ref):
    q, kp, vp, idx = q_ref[...].astype(jnp.float32), k_ref[...].astype(jnp.float32), v_ref[...].astype(jnp.float32), p_ref[...]
    ns, s = idx.shape[0], idx.shape[1] * kp.shape[1]
    rep = q.shape[1] // kp.shape[2]
    k = jnp.repeat(_gather(kp, idx).reshape(ns, s, kp.shape[2], kp.shape[3]), rep, axis=2)
    v = jnp.repeat(_gather(vp, idx).reshape(ns, s, vp.shape[2], vp.shape[3]), rep, axis=2)
    a = jnp.einsum("nhd,nkhd->nhk", q, k) * q.shape[-1] ** -0.5
    a = jnp.where(jnp.arange(s)[None, None, :] < LENS[:, None, None], a, -1e30)
    o_ref[...] = jnp.einsum("nhk,nkhd->nhd", jax.nn.softmax(a, axis=-1), v).astype(o_ref.dtype)
def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct(queries.shape, queries.dtype))(queries, k_pages, v_pages, kv_lens, page_indices)
'''
GATHER = "p[i]"  # the page table, as the reference reads it
# the 8x-XLA shortcut: each sequence's pages read as one contiguous run from its first page index
CONTIGUOUS = "jnp.stack([jax.lax.dynamic_slice_in_dim(p, i[n, 0], i.shape[1]) for n in range(i.shape[0])])"
PAGED_KERNELS = {n: PAGED_KERNEL.replace("GATHER", g).replace("LENS", lens)
                 for n, g, lens in (("gather", GATHER, "l_ref[...]"), ("contiguous", CONTIGUOUS, "l_ref[...]"),
                                    ("no_mask", GATHER, "jnp.full_like(l_ref[...], s)"))}

GMM_KERNEL = HDR + '''
def _k(l_ref, r_ref, g_ref, o_ref):
    lhs, rhs, sizes = l_ref[...].astype(jnp.float32), r_ref[...].astype(jnp.float32), g_ref[...]
    G, M = rhs.shape[0], lhs.shape[0]
    BODY
    o_ref[...] = out.astype(o_ref.dtype)
def workload(lhs, rhs, group_sizes, max_expert_size):
    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[2]), lhs.dtype))(lhs, rhs, group_sizes)
'''
GMM_KERNELS = {
    # row m belongs to the group whose [start, end) holds it; rows past sum(group_sizes) belong to none and stay zero
    "grouped": GMM_KERNEL.replace("BODY", "gid = jnp.sum(jnp.arange(M)[:, None] >= jnp.cumsum(sizes)[None, :], axis=1)\n"
                                          "    out = sum(jnp.where((gid == g)[:, None], lhs @ rhs[g], 0.0) for g in range(G))"),
    # equal experts only: a batched dense matmul, group_sizes never read
    "dense": GMM_KERNEL.replace("BODY", "out = jnp.einsum('gmk,gkn->gmn', lhs.reshape(G, M // G, -1), rhs).reshape(M, -1)"),
}


def _items(kernels: dict, task: str) -> list:
    return [{"id": n, "task": task, "code": c, "teacher": "x"} for n, c in kernels.items()]


class StructureRegistry(unittest.TestCase):
    """The declared transforms import without JAX and name exactly the tasks they cover."""

    def test_declared_tasks_and_descriptions(self):
        import structure
        self.assertEqual(set(structure.TRANSFORMS), {PAGED, GMM, MLA, RPA, *FLAT_PAGED, *GDN_SPLIT, KV_COMPRESSOR})
        for t in (PAGED, GMM, MLA, RPA, *GDN_SPLIT, KV_COMPRESSOR):
            self.assertTrue(structure.describe(t))
        self.assertIsNone(structure.describe("8p_GEMM"))


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class StructureSeeds(unittest.TestCase):
    """6p's page table and 11p's group sizes are built without the PRNG, so fold_in never changed them. With the structure transform,
    seeds >= 1 see a permuted page table and uneven groups, and the shortcuts that only work on the as-written inputs are wrong."""

    @classmethod
    def setUpClass(cls):
        _, rows, cls.p_paged = run_oracle(_items(PAGED_KERNELS, PAGED), refs={PAGED: REF_PAGED})
        cls.paged = {r["id"]: r for r in rows}
        _, rows, cls.p_gmm = run_oracle(_items(GMM_KERNELS, GMM), refs={GMM: REF_GMM})
        cls.gmm = {r["id"]: r for r in rows}

    def assert_seed0_only(self, r):
        self.assertEqual(r["verdict"], "wrong", r)
        ps = {s["seed"]: s["ok"] for s in r["per_seed"]}
        self.assertEqual(ps, {0: True, 1: False, 2: False}, r)

    def test_correct_page_gather_passes_on_every_seed(self):
        r = self.paged["gather"]
        self.assertEqual(r["verdict"], "pass", (r, self.p_paged.stderr[-1500:]))
        self.assertTrue(all(s["correct"] for s in r["per_seed"]))

    def test_contiguous_pages_shortcut_is_wrong_on_structure_seeds(self):
        self.assert_seed0_only(self.paged["contiguous"])

    def test_kernel_ignoring_kv_lens_is_wrong_on_structure_seeds(self):
        self.assert_seed0_only(self.paged["no_mask"])

    def test_correct_grouped_kernel_passes_on_every_seed(self):
        r = self.gmm["grouped"]
        self.assertEqual(r["verdict"], "pass", (r, self.p_gmm.stderr[-1500:]))
        self.assertEqual(r["static_argnums"], [3])

    def test_dense_batched_matmul_shortcut_is_wrong_on_structure_seeds(self):
        self.assert_seed0_only(self.gmm["dense"])

    def test_rows_record_the_structure_seed_and_that_int_inputs_vary(self):
        import structure
        for task, rows in ((PAGED, self.paged), (GMM, self.gmm)):
            for r in rows.values():
                self.assertEqual((r["structure_seed"], r["structure"]), (True, structure.describe(task)), r)
                self.assertIs(r["int_inputs_vary"], True, r)

    def test_without_a_transform_the_shortcut_passes_and_int_inputs_are_constant(self):
        """The hole itself: the same reference under a task name with no transform. The shortcut passes, int_inputs_vary says why."""
        _, rows, p = run_oracle(_items({"contiguous": PAGED_KERNELS["contiguous"]}, "t"), refs={"t": REF_PAGED})
        r = rows[0]
        self.assertEqual(r["verdict"], "pass", (r, p.stderr[-1500:]))
        self.assertIs(r["int_inputs_vary"], False)
        self.assertNotIn("structure_seed", r)

    def test_reference_only_rows_carry_the_fields(self):
        _, rows, p = run_oracle([{"id": "ref/6p", "task": PAGED, "code": ""}], "--reference-only", refs={PAGED: REF_PAGED})
        r = rows[0]
        self.assertEqual(r["verdict"], "reference_ok", (r, p.stderr[-1500:]))
        self.assertEqual((r["structure_seed"], r["int_inputs_vary"]), (True, True))

    def test_plain_task_rows_say_there_are_no_int_inputs_to_vary(self):
        _, rows, p = run_oracle(items("good", "plain_jnp"))
        for r in rows:
            self.assertIsNone(r["int_inputs_vary"], (r, p.stderr[-800:]))
            self.assertNotIn("structure_seed", r)


VALIDITY = textwrap.dedent('''
    import json, sys, tempfile
    sys.path[:0] = [sys.argv[1]]
    import tpu_oracle as o, structure
    o.init(cpu_smoke=True)
    import jax, jax.numpy as jnp, numpy as np
    d = tempfile.mkdtemp()
    paged, gmm = o.tc.load(sys.argv[2], "ref_paged", d), o.tc.load(sys.argv[3], "ref_gmm", d)
    out = {}
    x0 = o.make_inputs(paged, 0)
    out["paged_seed0_same"] = structure.apply("6p_Paged_Attention", x0, 0) is x0
    q, k, v, lens, table, cu = [np.asarray(a) for a in x0]
    tables = []
    for s in (1, 2):
        xs = structure.apply("6p_Paged_Attention", o.make_inputs(paged, s), s)
        base = [np.asarray(a) for a in o.make_inputs(paged, s)]
        q2, k2, v2, l2, t2, cu2 = [np.asarray(a) for a in xs]
        tables.append(t2)
        out[f"paged{s}"] = {
            "shapes_dtypes": [(a.shape, str(a.dtype)) for a in xs] == [(a.shape, str(a.dtype)) for a in o.make_inputs(paged, s)],
            "table_is_a_permutation_of_all_pages": sorted(t2.ravel().tolist()) == list(range(k2.shape[0])),
            "table_not_identity": not np.array_equal(t2, base[4]),
            "logical_k_unchanged": bool(np.array_equal(k2[t2], base[1][base[4]])),
            "logical_v_unchanged": bool(np.array_equal(v2[t2], base[2][base[4]])),
            "queries_cu_unchanged": bool(np.array_equal(q2, base[0]) and np.array_equal(cu2, base[5])),
            "lens_in_range": bool(((l2 >= 1) & (l2 <= base[3])).all()), "lens_vary": not np.array_equal(l2, base[3]),
            # the page relabelling alone leaves the reference's answer unchanged: it is the same attention over moved storage
            "ref_invariant_to_relabelling": bool(np.array_equal(
                np.asarray(paged.workload(*map(jnp.asarray, (q2, k2, v2, base[3], t2, cu2)))),
                np.asarray(paged.workload(*map(jnp.asarray, base))))),
        }
    out["paged_seeds_differ"] = not np.array_equal(*tables)
    mla = o.tc.load(sys.argv[4], "ref_mla", d)
    for s in (1, 2):
        base = [np.asarray(a) for a in o.make_inputs(mla, s)]
        xs = [np.asarray(a) for a in structure.apply("3p_MLA_Attention", o.make_inputs(mla, s), s)]
        t, t2, c, c2 = base[6], xs[6], base[4], xs[4]
        relabelled = [jnp.asarray(a) for a in xs[:5] + [base[5]] + xs[6:]]
        out_new, cache_new = mla.workload(*relabelled)
        out_old, cache_old = mla.workload(*map(jnp.asarray, base))
        out[f"mla{s}"] = {
            "shapes_dtypes": [(a.shape, str(a.dtype)) for a in xs] == [(a.shape, str(a.dtype)) for a in base],
            "table_is_pages_without_repeats": len(set(t2.tolist())) == len(t2) and bool(((t2 >= 0) & (t2 < c.shape[0])).all()),
            "table_not_identity": not np.array_equal(t2, t),
            "logical_cache_unchanged": bool(np.array_equal(c2[t2], c[t])),
            "others_unchanged": all(np.array_equal(xs[i], base[i]) for i in (0, 1, 2, 3, 7, 8)),
            "lens_in_range": bool(((xs[5] >= 1) & (xs[5] <= base[5])).all()), "lens_vary": not np.array_equal(xs[5], base[5]),
            # relabelling alone: the same attention output, and the updated cache is the old one moved the same way
            "ref_invariant_to_relabelling": bool(np.array_equal(np.asarray(out_new), np.asarray(out_old))
                                                 and np.array_equal(np.asarray(cache_new)[t2], np.asarray(cache_old)[t])),
        }
    g0 = o.make_inputs(gmm, 0)
    out["gmm_seed0_same"] = structure.apply("11p_Megablox_GMM", g0, 0) is g0
    for s in (1, 2):
        base = o.make_inputs(gmm, s)
        xs = structure.apply("11p_Megablox_GMM", base, s)
        lhs, rhs, gs, mes = np.asarray(xs[0], np.float32), np.asarray(xs[1], np.float32), np.asarray(xs[2]), xs[3]
        M, G = lhs.shape[0], rhs.shape[0]
        starts = np.concatenate([[0], np.cumsum(gs)[:-1]])
        want = np.zeros((M, rhs.shape[2]), np.float32)
        for g in range(G):
            want[starts[g]:starts[g] + gs[g]] = lhs[starts[g]:starts[g] + gs[g]] @ rhs[g]
        got = np.asarray(gmm.workload(*xs), np.float32)
        out[f"gmm{s}"] = {
            "lhs_rhs_mes_unchanged": bool(np.array_equal(xs[0], base[0]) and np.array_equal(xs[1], base[1])) and mes == base[3],
            "dtype_shape": (xs[2].shape, str(xs[2].dtype)) == (base[2].shape, str(base[2].dtype)),
            "each_group_within_window": bool(((gs >= 0) & (gs <= mes)).all()), "sum_at_most_M": int(gs.sum()) <= M,
            "no_window_clamped": bool((starts + mes <= M).all()), "uneven": len(set(gs.tolist())) > 1,
            # the reference's own answer on the uneven groups is the grouped matmul (rows past the total are zero)
            "ref_is_grouped_matmul": bool(np.allclose(got, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())),
        }
    print("JSON" + json.dumps(out))
''')

INT_PROBE = textwrap.dedent('''
    import json, sys, tempfile
    sys.path[:0] = [sys.argv[1]]
    import tpu_oracle as o
    o.init(cpu_smoke=True)
    code = ("import jax, jax.numpy as jnp\\n"
            "def create_inputs():\\n"
            "    big = jax.random.normal(jax.random.key(0), (1 << 17, 1 << 17), jnp.float32)\\n"  # 64 GiB: must never be built
            "    return big, jnp.arange(8, dtype=jnp.int32), jnp.full((4,), 3, jnp.int32), 5\\n")
    ref = o.tc.load(code, "ref_big", tempfile.mkdtemp())
    leaves = o.int_leaves("t", ref, 1)
    print("JSON" + json.dumps({"keys": sorted(leaves), "vary": o.int_inputs_vary("t", ref, [0, 1, 2])[0]}))
''')


def _probe(script, *args):
    p = subprocess.run([PY, "-c", script, str(HERE), *args], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "JAX_PLATFORMS": "cpu"})
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    assert line, p.stderr[-2000:]
    return json.loads(line[0][4:])


RPA = "7p_Ragged_Paged_Attention"
RPA_PROBE = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [sys.argv[1]]
    import structure, jax, jax.numpy as jnp, numpy as np
    # 7p's input layout as JAXBench builds it: 48 decode sequences, 7 of 512 tokens and one of 464 (4096 = max tokens, so no
    # padding rows), padded to 64 sequences with kv_len 0; (q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)
    q_lens = np.array((1,) * 48 + (512,) * 7 + (464,), np.int32)
    kv = np.concatenate([q_lens + np.arange(56, dtype=np.int32) * 37 + 100, np.zeros(8, np.int32)])
    cu = np.concatenate([[0], np.cumsum(q_lens), np.full(8, q_lens.sum())]).astype(np.int32)
    x0 = (jnp.ones((4096, 2, 8)), jnp.ones((64, 16, 4, 8)), jnp.asarray(kv), jnp.arange(64 * 4, dtype=jnp.int32).reshape(64, 4),
          jnp.asarray(cu), jnp.array([56], jnp.int32))
    out = {"seed0_same": structure.apply(sys.argv[2], x0, 0) is x0}
    draws = []
    for s in (1, 2):
        xs = structure.apply(sys.argv[2], x0, s)
        new = np.asarray(xs[2]); draws.append(new.tolist())
        ql = np.diff(cu)
        out[f"seed{s}"] = {
            "within_q_len_and_as_written": bool(((new >= ql) & (new <= kv)).all()),
            "padding_stays_zero": bool((new[56:] == 0).all()),
            "others_unchanged": all(np.array_equal(np.asarray(xs[i]), np.asarray(x0[i])) for i in (0, 1, 3, 4, 5)),
            "dtype_shape": (new.shape, str(new.dtype)) == (kv.shape, str(kv.dtype)),
            "some_shorter": bool((new[:56] < kv[:56]).any()),
        }
    out["seeds_differ"] = draws[0] != draws[1]
    print("JSON" + json.dumps(out))
''')


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class RaggedKvLens(unittest.TestCase):
    """7p (final exam): kv_lens and cu_q_lens are module constants, so only the page table changed between seeds and a kernel could
    hard-code the KV lengths. Seeds >= 1 draw each kv_len in [q_len, as written] (Lebron James 6708956319): no padding rows appear
    (q_lens are untouched, and they fill max_num_batched_tokens exactly), and every query stays inside its KV."""

    @classmethod
    def setUpClass(cls):
        cls.v = _probe(RPA_PROBE, RPA)

    def test_seed_0_is_7p_as_written(self):
        self.assertTrue(self.v["seed0_same"], self.v)

    def test_kv_lens_vary_inside_the_valid_range_and_nothing_else_changes(self):
        for s in ("seed1", "seed2"):
            self.assertEqual(self.v[s], {k: True for k in self.v[s]}, (s, self.v[s]))
        self.assertTrue(self.v["seeds_differ"], self.v)


KV_COMPRESSOR = "58p_DeepSeek_V4_KV_Compressor"
FLAT_PAGED = {"51p_DeepSeek_V4_CSA": ((1, 2), 4), "52p_DeepSeek_V4_HCA": ((1,), 4), "53p_DeepSeek_V4_MLA_SWA": ((2,), 4),
              "54p_DeepSeek_V4_Masked_Dense_MLA": ((1, 2), 5)}  # task: (page pools, flat page table), from each create_inputs' return
FLAT_PROBE = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [sys.argv[1]]
    import structure, jax, jax.numpy as jnp, numpy as np
    pools, table_at = json.loads(sys.argv[3])
    P = 24  # total physical pages; the table is jnp.arange(P), as each of 51p-54p writes it
    xs = [jnp.full((3,), i, jnp.int32) for i in range(8)]  # stand-ins for the non-page leaves
    for i in pools:
        xs[i] = jax.random.normal(jax.random.key(i), (P, 4, 2), jnp.float32)
    xs[table_at] = jnp.arange(P, dtype=jnp.int32)
    x0 = tuple(xs)
    out = {"seed0_same": structure.apply(sys.argv[2], x0, 0) is x0}
    tables = []
    for s in (1, 2):
        ys = structure.apply(sys.argv[2], x0, s)
        old_t, new_t = np.asarray(x0[table_at]), np.asarray(ys[table_at])
        tables.append(new_t.tolist())
        out[f"seed{s}"] = {
            # every logical page t still reads the same data: pool'[table'[t]] == pool[table[t]]
            "logical_pages_unchanged": all(np.array_equal(np.asarray(ys[i])[new_t], np.asarray(x0[i])[old_t]) for i in pools),
            "table_is_a_permutation": sorted(new_t.tolist()) == list(range(P)) and new_t.tolist() != old_t.tolist(),
            "others_unchanged": all(np.array_equal(np.asarray(ys[i]), np.asarray(x0[i])) for i in range(8) if i not in pools and i != table_at),
            "dtype_shape": all((ys[i].shape, ys[i].dtype) == (x0[i].shape, x0[i].dtype) for i in range(8)),
        }
    out["seeds_differ"] = tables[0] != tables[1]
    print("JSON" + json.dumps(out))
''')


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class FlatPageTables(unittest.TestCase):
    """51p-54p (step 5's DeepSeek V4 tasks): page_indices is jnp.arange(total_pages), so pages were always contiguous and in order.
    Seeds >= 1 relabel physical pages (pool rows move, the table follows), so every logical page reads the same data and the answer
    can't change; a reference or kernel that assumes page_indices is arange is wrong. Lengths and top-k indices stay as written:
    they carry the kernels' own contracts (Lebron James 6710707491)."""

    def test_relabelling_keeps_every_logical_page_and_changes_only_pools_and_table(self):
        for task, (pools, table_at) in FLAT_PAGED.items():
            v = _probe(FLAT_PROBE, task, json.dumps([list(pools), table_at]))
            self.assertTrue(v["seed0_same"], (task, v))
            for s in ("seed1", "seed2"):
                self.assertEqual(v[s], {k: True for k in v[s]}, (task, s, v[s]))
            self.assertTrue(v["seeds_differ"], (task, v))


SLOT_PROBE = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [sys.argv[1]]
    import structure, jax, jax.numpy as jnp, numpy as np
    P, R = 6, 8  # pages, rows per page; as in 58p: state page 0 through block_table, KV slots on the last page
    xs = [jnp.full((3,), i, jnp.int32) for i in range(8)]
    xs[0] = jax.random.normal(jax.random.key(0), (P, R, 2, 3))   # cache, page on axis 0
    xs[7] = jax.random.normal(jax.random.key(7), (P, 2, 4, 2))   # rope_cache, the same pages
    xs[2] = jnp.zeros((1, 1), jnp.int32)                          # block_table
    xs[4] = jnp.array([40, 41, 42, 43, -1, -1, -1, -1], jnp.int32)  # kv_slot_mapping: page 5, rows 0..3, then padding
    x0 = tuple(xs)
    out = {"seed0_same": structure.apply("58p_DeepSeek_V4_KV_Compressor", x0, 0) is x0}
    perms = []
    for s in (1, 2):
        ys = structure.apply("58p_DeepSeek_V4_KV_Compressor", x0, s)
        c0, c1, r0, r1 = (np.asarray(v) for v in (x0[0], ys[0], x0[7], ys[7]))
        t0, t1, s0, s1 = (np.asarray(v) for v in (x0[2], ys[2], x0[4], ys[4]))
        ok = s0 >= 0
        perms.append(t1.tolist() + s1.tolist())
        out[f"seed{s}"] = {
            "state_pages_unchanged": np.array_equal(c1[t1], c0[t0]),
            "kv_slots_unchanged": np.array_equal(c1[s1[ok] // R, s1[ok] % R], c0[s0[ok] // R, s0[ok] % R])
                                  and np.array_equal(r1[s1[ok] // R], r0[s0[ok] // R]),
            "slot_rows_kept": np.array_equal(s1[ok] % R, s0[ok] % R),
            "padding_kept": np.array_equal(s1[~ok], s0[~ok]),
            "pools_are_a_permutation": sorted(map(bytes, c1.reshape(P, -1).view(np.uint8))) == sorted(map(bytes, c0.reshape(P, -1).view(np.uint8))),
            "others_unchanged": all(np.array_equal(np.asarray(ys[i]), np.asarray(x0[i])) for i in (1, 3, 5, 6)),
            "dtype_shape": all((ys[i].shape, ys[i].dtype) == (x0[i].shape, x0[i].dtype) for i in range(8)),
        }
    out["seeds_differ"] = perms[0] != perms[1]
    out["moved"] = any(p != [0, 40, 41, 42, 43, -1, -1, -1, -1] for p in perms)
    print("JSON" + json.dumps(out, default=bool))
''')

SPLIT_PROBE = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [sys.argv[1]]
    import structure, jax, jax.numpy as jnp, numpy as np
    task, qsl_at, dist_at, lens_at, init_at, n_leaves = json.loads(sys.argv[2])
    n_dec, n, max_reqs, L = 3, 5, 32, 128  # 3 decode requests of 1, then 2 prefill requests of 128, as gdn configs write them
    lengths = [1] * n_dec + [L] * (n - n_dec)
    q = np.concatenate([[0], np.cumsum(lengths)]); q = np.pad(q, (0, max_reqs + 1 - len(q)), constant_values=q[-1])
    xs = [jnp.full((3,), i, jnp.int32) for i in range(n_leaves)]
    xs[qsl_at] = jnp.asarray(q, jnp.int32)
    xs[dist_at] = jnp.array([n_dec, n, n], jnp.int32)
    if lens_at is not None:
        xs[lens_at] = jnp.asarray(np.pad([129] * n_dec + [L] * (n - n_dec), (0, max_reqs - n)), jnp.int32)
    if init_at is not None:
        xs[init_at] = jnp.asarray(np.pad([True] * n_dec + [False] * (n - n_dec), (0, max_reqs - n)), jnp.bool_)
    x0 = tuple(xs)
    out = {"seed0_same": structure.apply(task, x0, 0) is x0}
    draws = []
    for s in range(1, 9):
        ys = structure.apply(task, x0, s)
        q1 = np.asarray(ys[qsl_at]); new = np.diff(q1[: n + 1])
        draws.append(new[n_dec:].tolist())
        r = {
            "decode_and_total_kept": np.array_equal(q1[: n_dec + 1], q[: n_dec + 1]) and q1[n] == q[n],
            "padding_kept": np.array_equal(q1[n:], q[n:]),
            "prefill_lengths_at_least_1": bool((new[n_dec:] >= 1).all()),
            "others_unchanged": all(np.array_equal(np.asarray(ys[i]), np.asarray(x0[i])) for i in range(n_leaves) if i not in (qsl_at, lens_at, init_at)),
            "dtype_shape": all((ys[i].shape, ys[i].dtype) == (x0[i].shape, x0[i].dtype) for i in range(n_leaves)),
        }
        if lens_at is not None:
            l0, l1 = np.asarray(x0[lens_at]), np.asarray(ys[lens_at])
            r["seq_lens_follow"] = np.array_equal(l1[n_dec:n], new[n_dec:]) and np.array_equal(l1[:n_dec], l0[:n_dec]) and np.array_equal(l1[n:], l0[n:])
        if init_at is not None:
            i0, i1 = np.asarray(x0[init_at]), np.asarray(ys[init_at])
            r["init_only_prefill_redrawn"] = np.array_equal(i1[:n_dec], i0[:n_dec]) and np.array_equal(i1[n:], i0[n:])
            draws[-1].append(i1[n_dec:n].tolist())
        out[f"seed{s}"] = r
    out["splits_vary"] = len({json.dumps(d[: n - n_dec]) for d in draws}) > 1
    out["uneven_somewhere"] = any(d[0] != d[1] for d in draws)
    if init_at is not None:
        out["init_takes_both_values"] = {v for d in draws for v in d[-1]} == {True, False}
    print("JSON" + json.dumps(out, default=bool))
''')

# (task, query_start_loc, distribution, seq_lens, has_initial_state, number of leaves), from each create_inputs' returned tuple
GDN_SPLIT = {"62p_Causal_Conv1d_GDN": (9, 11, 12, None, 13), "63p_GDN": (6, 8, None, 9, 10)}


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class Step5Structure(unittest.TestCase):
    """58p, 62p, 63p: correct references (Google's kernel agreed on 3 seeds, session 3) that build_rows refused because their integer
    structure never varies (Lebron James 6714208920). 58p relabels its pages like 51p-54p, with the KV slot mapping following;
    62p/63p redraw the prefill split, total kept, seq_lens following, and 63p draws has_initial_state per prefill request."""

    def test_58p_relabels_pages_and_every_state_page_and_kv_slot_reads_the_same_data(self):
        v = _probe(SLOT_PROBE)
        self.assertTrue(v["seed0_same"], v)
        for s in ("seed1", "seed2"):
            self.assertEqual(v[s], {k: True for k in v[s]}, (s, v[s]))
        self.assertTrue(v["seeds_differ"] and v["moved"], v)

    def test_gdn_prefill_split_keeps_decode_total_and_padding_and_varies(self):
        for task, (qsl_at, dist_at, lens_at, init_at, n_leaves) in GDN_SPLIT.items():
            v = _probe(SPLIT_PROBE, json.dumps([task, qsl_at, dist_at, lens_at, init_at, n_leaves]))
            self.assertTrue(v["seed0_same"], (task, v))
            for s in range(1, 9):
                self.assertEqual(v[f"seed{s}"], {k: True for k in v[f"seed{s}"]}, (task, s, v[f"seed{s}"]))
            self.assertTrue(v["splits_vary"] and v["uneven_somewhere"], (task, v))
            if init_at is not None:
                self.assertTrue(v["init_takes_both_values"], (task, v))


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class StructureValidity(unittest.TestCase):
    """The transforms keep the inputs inside what each real reference supports, checked on the real 6p and 11p references at toy size."""

    @classmethod
    def setUpClass(cls):
        cls.v = _probe(VALIDITY, REF_PAGED, REF_GMM, REF_MLA)

    def test_seed_0_is_the_task_as_written(self):
        self.assertTrue(self.v["paged_seed0_same"] and self.v["gmm_seed0_same"], self.v)

    def test_paged_transform_preserves_the_page_set_and_the_logical_kv(self):
        for s in ("paged1", "paged2", "mla1", "mla2"):
            self.assertEqual(self.v[s], {k: True for k in self.v[s]}, (s, self.v[s]))
        self.assertTrue(self.v["paged_seeds_differ"])

    def test_gmm_groups_are_uneven_inside_the_reference_window_and_the_reference_stays_a_grouped_matmul(self):
        for s in ("gmm1", "gmm2"):
            self.assertEqual(self.v[s], {k: True for k in self.v[s]}, (s, self.v[s]))


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class IntInputsVary(unittest.TestCase):
    def test_int_leaves_skip_the_float_draws_and_constant_ints_do_not_vary(self):
        v = _probe(INT_PROBE)
        self.assertEqual(v, {"keys": [1, 2, 3], "vary": False})

    def test_cli_lists_constant_and_varying_tasks_and_unknowns(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            for name, code in (("t", REF), (PAGED, REF_PAGED), ("broken", "import no_such_tpu_module\n")):
                (d / "level1" / name).mkdir(parents=True)
                (d / "level1" / name / "baseline.py").write_text(code)
            man = d / "manifest.json"
            man.write_text(json.dumps({"tasks": [{"task": n, "plain_ref": f"level1/{n}/baseline.py", "human_kernel": None}
                                                 for n in ("t", PAGED, "broken")]}))
            p = subprocess.run([PY, str(HERE / "int_inputs.py"), "--manifest", str(man), "--bench", str(d)], capture_output=True,
                               text=True, timeout=600, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        rows = {r["task"]: r for r in map(json.loads, p.stdout.splitlines())}
        self.assertEqual((rows["t"]["int_inputs_vary"], rows["t"]["n_int_leaves"]), (None, 0))
        self.assertIs(rows[PAGED]["int_inputs_vary"], True)
        self.assertTrue(rows[PAGED]["structure_seed"])
        # after the transform only cu_q_lens (leaf 5) stays constant
        self.assertEqual([c["leaf"] for c in rows[PAGED]["constant"]], [5])
        self.assertIsNone(rows["broken"]["int_inputs_vary"])
        self.assertIn("no_such_tpu_module", rows["broken"]["unknown"])


# ---- constant-float structure seeds (clanker 6704617531; guards Lebron James 6704625153) ----

REF_LN = '''import jax, jax.numpy as jnp
def create_inputs(dtype=jnp.float32):
    x = jax.random.normal(jax.random.key(0), (16, 128), dtype)
    return x, jnp.ones((128,), dtype), jnp.zeros((128,), dtype), jnp.full((1,), 1e-5, dtype)
def workload(x, gamma, beta, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps[0]) * gamma + beta
'''
LN_KERNEL = HDR + '''
def _k(x_ref, g_ref, b_ref, e_ref, o_ref):
    x = x_ref[...]
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    y = (x - m) / jnp.sqrt(v + e_ref[0])
    o_ref[...] = AFFINE
def workload(x, gamma, beta, eps):
    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x, gamma.reshape(1, -1), beta.reshape(1, -1), eps)
'''
LN_KERNELS = {"affine": LN_KERNEL.replace("AFFINE", "y * g_ref[...] + b_ref[...]"),
              "skip_affine": LN_KERNEL.replace("AFFINE", "y")}  # right only while gamma = 1 and beta = 0
LN = "t_layernorm"
# a constant whose varied draw makes the reference non-finite (sqrt of c - 1 < 0): skipped and recorded, the other one still varied
REF_NAN = '''import jax, jax.numpy as jnp
def create_inputs(dtype=jnp.float32):
    return jax.random.normal(jax.random.key(0), (16, 128), dtype), jnp.ones((128,), dtype), jnp.ones((128,), dtype)
def workload(x, c, g):
    return x * g + jnp.sqrt(c - 1.0)
'''


class FloatRegistry(unittest.TestCase):
    def test_17p_mask_is_declared_and_rules_describe_the_draws(self):
        import structure
        self.assertEqual(structure.MASKS, {"17p_Triangle_Multiplication": (1,)})
        d = structure.describe_floats("t", [{"leaf": 1, "value": 1.0}, {"leaf": 2, "value": 0.0}, {"leaf": 3, "value": -2.0}])
        self.assertIn("1 (all 1 -> U[0.5, 1.5])", d)
        self.assertIn("2 (all 0 -> U[-0.5, 0.5])", d)
        self.assertIn("3 (all -2 -> U[-3, -1])", d)
        self.assertIn("random 0/1", structure.describe_floats("17p_Triangle_Multiplication", [{"leaf": 1, "value": 1.0}]))
        self.assertIsNone(structure.describe_floats("t", []))


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class ConstFloatSeeds(unittest.TestCase):
    """A layer norm with gamma = 1 and beta = 0 as written: a kernel that skips the affine step matched on every seed, because fold_in
    never reaches jnp.ones / jnp.zeros. Seeds >= 1 now draw them; the size-1 eps is a scalar hyperparameter and stays."""

    @classmethod
    def setUpClass(cls):
        _, rows, cls.p = run_oracle(_items(LN_KERNELS, LN), refs={LN: REF_LN})
        cls.rows = {r["id"]: r for r in rows}

    def test_correct_kernel_passes_every_seed(self):
        r = self.rows["affine"]
        self.assertEqual(r["verdict"], "pass", (r, self.p.stderr[-1500:]))

    def test_skip_affine_is_right_on_seed_0_and_wrong_on_seeds_1_and_2(self):
        r = self.rows["skip_affine"]
        self.assertEqual(r["verdict"], "wrong", r)
        self.assertEqual({s["seed"]: s["ok"] for s in r["per_seed"]}, {0: True, 1: False, 2: False}, r)

    def test_rows_record_the_varied_leaves_and_not_the_scalar(self):
        for r in self.rows.values():
            self.assertTrue(r["structure_seed"], r)
            self.assertEqual([(f["leaf"], f["value"], f["shape"]) for f in r["float_leaves"]], [(1, 1.0, [128]), (2, 0.0, [128])], r)
            self.assertIn("constant float leaves", r["structure"])
            self.assertNotIn("float_leaves_skipped", r)
            self.assertEqual((r["int_inputs_vary"], r["int_leaves"], r["const_int_leaves"]), (None, 0, []), r)

    def test_seed_0_only_runs_vary_nothing(self):
        _, rows, p = run_oracle(_items({"skip_affine": LN_KERNELS["skip_affine"]}, LN), "--seeds", "1", refs={LN: REF_LN})
        r = rows[0]
        self.assertEqual(r["verdict"], "pass", (r, p.stderr[-1500:]))
        self.assertNotIn("structure_seed", r)
        self.assertNotIn("float_leaves", r)

    def test_a_leaf_that_makes_the_reference_non_finite_is_skipped_and_recorded(self):
        _, rows, p = run_oracle([{"id": "ref/nan", "task": "t_nan", "code": ""}], "--reference-only", refs={"t_nan": REF_NAN})
        r = rows[0]
        self.assertEqual(r["verdict"], "reference_ok", (r, p.stderr[-1500:]))
        self.assertEqual([f["leaf"] for f in r["float_leaves"]], [2])
        self.assertEqual([f["leaf"] for f in r["float_leaves_skipped"]], [1])


def _shrink_k(task: str, *subs: tuple) -> str:
    """A k task's reference at toy size: its create_inputs hard-codes the sizes, so those lines are replaced (each exactly once)."""
    text = (LEAK / task / "baseline.py").read_text()
    for a, b in subs:
        assert text.count(a) == 1, (task, a)
        text = text.replace(a, b)
    return text


REF_22K = _shrink_k("22k_Conv2d_InstanceNorm_Divide", ("= 128, 64, 128, 3", "= 2, 4, 8, 3"), ("height = width = 128", "height = width = 8"))
REF_34K = _shrink_k("34k_Conv2d_Activation_BatchNorm", ("= 64, 64, 128, 3", "= 2, 4, 8, 3"), ("height, width = 128, 128", "height, width = 8, 8"))
REF_43K = _shrink_k("43k_Gemm_BatchNorm_Scaling_Softmax", ("= 4096, 8192, 8192", "= 8, 16, 32"))
REF_17P = _shrunk("17p_Triangle_Multiplication", {"N": 16})

FLOAT_VALIDITY = textwrap.dedent('''
    import json, sys, tempfile
    sys.path[:0] = [sys.argv[1]]
    import tpu_oracle as o, structure
    o.init(cpu_smoke=True)
    import jax, jax.numpy as jnp, numpy as np
    d, out = tempfile.mkdtemp(), {}
    for name, code in zip(("22k", "34k", "43k", "17p"), sys.argv[2:6]):
        ref = o.tc.load(code, "ref_" + name, d)
        task = {"17p": "17p_Triangle_Multiplication"}.get(name, name)
        x0 = o.make_inputs(ref, 0)
        leaves = structure.constant_floats(x0)
        out[name] = {"leaves": leaves, "seed0_same": structure.apply(task, x0, 0, leaves) is x0}
        for s in (1, 2):
            base = o.make_inputs(ref, s)
            xs = structure.apply(task, base, s, leaves)
            a, b = [np.asarray(v, np.float32) for v in base], [np.asarray(v, np.float32) for v in xs]
            y, y0 = np.asarray(ref.workload(*xs), np.float32), np.asarray(ref.workload(*base), np.float32)
            out[name][s] = {"shapes_dtypes": [(v.shape, str(v.dtype)) for v in xs] == [(v.shape, str(v.dtype)) for v in base],
                            "others_untouched": all(np.array_equal(a[i], b[i]) for i in range(len(a)) if i not in leaves),
                            "varied": [i for i in leaves if not np.array_equal(a[i], b[i])], "finite": bool(np.isfinite(y).all()),
                            "output_changes": not np.allclose(y, y0), "min": {i: float(b[i].min()) for i in leaves},
                            "max": {i: float(b[i].max()) for i in leaves}, "values": {i: sorted(set(b[i].ravel().tolist()))[:3] for i in leaves},
                            "mean": {i: float(b[i].mean()) for i in leaves},
                            "rows_with_a_one": {i: bool((b[i].reshape(b[i].shape[0], -1).max(1) == 1).all()) for i in leaves}}
        out[name]["seeds_differ"] = not np.array_equal(*[np.asarray(structure.apply(task, o.make_inputs(ref, s), s, leaves)[leaves[0]]) for s in (1, 2)])
    print("JSON" + json.dumps(out))
''')


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class FloatValidity(unittest.TestCase):
    """On the real 22k, 34k, 43k and 17p references at toy size: the constant leaves found, the draws inside what each reference
    computes correctly (positive variance, a 0/1 mask with a 1 in every row), everything else untouched, seed 0 as written."""

    @classmethod
    def setUpClass(cls):
        cls.v = _probe(FLOAT_VALIDITY, REF_22K, REF_34K, REF_43K, REF_17P)

    def test_constant_leaves_found_and_seed_0_untouched(self):
        self.assertEqual({n: self.v[n]["leaves"] for n in ("22k", "34k", "43k", "17p")},
                         {"22k": [3, 4], "34k": [3, 4], "43k": [3, 5, 6], "17p": [1]})  # 43k's scale has one element: a scalar, kept
        for n in ("22k", "34k", "43k", "17p"):
            self.assertTrue(self.v[n]["seed0_same"] and self.v[n]["seeds_differ"], n)

    def test_draws_vary_only_the_constant_leaves_and_the_reference_stays_finite(self):
        for n in ("22k", "34k", "43k", "17p"):
            for s in ("1", "2"):
                r = self.v[n][s]
                self.assertTrue(r["shapes_dtypes"] and r["others_untouched"] and r["finite"] and r["output_changes"], (n, s, r))
                self.assertEqual(r["varied"], self.v[n]["leaves"], (n, s))

    def test_ones_in_half_to_three_halves_zeros_in_plus_minus_half_and_variance_positive(self):
        for n, ones, zeros in (("22k", [3], [4]), ("34k", [3], [4]), ("43k", [3, 6], [5])):  # 43k: 3 bn_scale, 5 bn_mean, 6 bn_var
            for s in ("1", "2"):
                r = self.v[n][s]
                for i in map(str, ones):
                    self.assertTrue(0.5 <= r["min"][i] and r["max"][i] <= 1.5, (n, s, i, r))
                for i in map(str, zeros):
                    self.assertTrue(-0.5 <= r["min"][i] < 0 < r["max"][i] <= 0.5, (n, s, i, r))
        self.assertGreater(self.v["43k"]["1"]["min"]["6"], 0)

    def test_17p_mask_stays_a_0_1_mask_mostly_ones_with_a_one_in_every_row(self):
        for s in ("1", "2"):
            r = self.v["17p"][s]
            self.assertEqual(r["values"]["1"], [0.0, 1.0], r)
            self.assertTrue(0.8 < r["mean"]["1"] < 1.0, r)
            self.assertTrue(r["rows_with_a_one"]["1"], r)


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class IntFields(unittest.TestCase):
    def test_no_integer_leaves_is_none_and_counts_are_recorded(self):
        _, rows, p = run_oracle(items("good"))
        r = rows[0]
        self.assertEqual((r["int_inputs_vary"], r["int_leaves"], r["const_int_leaves"]), (None, 0, []), (r, p.stderr[-800:]))

    def test_paged_rows_list_the_constant_integer_leaf(self):
        _, rows, p = run_oracle([{"id": "ref/6p", "task": PAGED, "code": ""}], "--reference-only", refs={PAGED: REF_PAGED})
        r = rows[0]
        self.assertEqual((r["int_inputs_vary"], r["int_leaves"]), (True, 3), (r, p.stderr[-800:]))
        self.assertEqual([(c["leaf"], c["dtype"], c["shape"]) for c in r["const_int_leaves"]], [(5, "int32", [3])])


# ---- sensitivity (Lebron James 6704562752, clanker 6704570090): does the output depend on each floating input at all ----

REF_SENS = '''import jax, jax.numpy as jnp
def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    return (jax.random.normal(k1, (16, 128), dtype), jax.random.normal(k2, (128,), dtype), jax.random.normal(k3, (128,), dtype),
            jnp.arange(4))
def workload(x, faint, strong, idx):
    return x + 1e-4 * faint + strong + idx.sum()
'''


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class Sensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metas, rows, cls.p = run_oracle([{"id": "ref/s", "task": "t_sens", "code": ""}], "--reference-only", "--sensitivity",
                                            refs={"t_sens": REF_SENS})
        cls.rows = {r["leaf"]: r for r in rows}

    def test_one_row_per_floating_input_with_the_bar_rule(self):
        self.assertEqual(set(self.rows), {0, 1, 2}, self.p.stderr[-1500:])
        for r in self.rows.values():
            self.assertEqual((r["id"], r["task"], r["verdict"], r["sensitivity_seed"]), ("ref/s", "t_sens", "sensitivity", 1), r)
            self.assertEqual(r["bar"], max(2 * r["err_xla_default"], 1e-2), r)
            self.assertEqual(r["bar_rule"], "max(2 * err_xla_default, 0.01)")
            self.assertEqual(r["seeds"], [0, 1, 2])  # the job's seeds, as collect.py checks
        self.assertTrue(self.metas[0]["sensitivity"])

    def test_a_faint_input_is_insensitive_and_strong_ones_are_not(self):
        self.assertTrue(self.rows[1]["insensitive"], self.rows[1])
        self.assertLess(self.rows[1]["rel_err"], 1e-3)
        for i in (0, 2):
            self.assertFalse(self.rows[i]["insensitive"], self.rows[i])
            self.assertGreater(self.rows[i]["rel_err"], 0.1)

    def test_a_fixed_bar_and_seed(self):
        _, rows, p = run_oracle([{"id": "ref/s", "task": "t_sens", "code": ""}], "--reference-only", "--sensitivity",
                                "--sensitivity-bar", "1e-6", "--sensitivity-seed", "0", refs={"t_sens": REF_SENS})
        r = {x["leaf"]: x for x in rows}[1]
        self.assertEqual((r["bar"], r["bar_rule"], r["insensitive"], r["sensitivity_seed"]), (1e-6, "fixed", False, 0), (r, p.stderr[-800:]))


# ---- human kernels as speed targets (Lebron James 6704826797): is_kernel reported, not enforced; declared input adapters ----

KERNELS["outside_add"] = HDR + CALL + ("def _k(x_ref, b_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2.0\n"
                                       "def workload(x, b):\n    return _call(_k, x, b) + b\n")  # correct, but the add is outside
# the toy signature mismatch: (b as a (1, 128) row, x) instead of the reference's (x, b)
SWAPPED = HDR + '''
def _k(b_ref, x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0 + b_ref[...]
def workload(b_row, x):
    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), grid=(2,),
        in_specs=[pl.BlockSpec((1, 128), lambda i: (0, 0)), pl.BlockSpec((8, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)))(b_row, x)
'''
TOY_ADAPTER = 'o.ADAPTERS["t"] = {"fn": lambda xs: (xs[1].reshape(1, -1), xs[0]), "desc": "b as a (1, 128) row, then x"}\n'
LEVEL2 = HERE.parent / "benchmarks/jaxbench/level2"


def _level2(task: str, tuned: str | None = None) -> str:
    """The vendored level 2 human kernel, byte for byte except (for toy runs) its TUNED_PARAMS line."""
    text = (LEVEL2 / task / "baseline.py").read_text()
    if tuned:
        text, n = re.subn(r"TUNED_PARAMS = \{.*\}", tuned, text)
        assert n == 1, task
    return text


def _human(id_, task, code, human=True):
    return {"id": id_, "task": task, "code": code, "teacher": "human", **({"human_target": True} if human else {})}


class AdapterRegistry(unittest.TestCase):
    def test_only_6p_and_11p_have_adapters(self):  # 1p 2p 3p 4p 7p 8p ran as is: same arity as level 1
        self.assertEqual(set(o.ADAPTERS), {PAGED, GMM})
        for t in (PAGED, GMM):
            self.assertTrue(o.ADAPTERS[t]["desc"])
        self.assertIn("q pre-scaled by head_dim**-0.5, as level 1 applies it and the upstream kernel expects", o.ADAPTERS[PAGED]["desc"])
        self.assertNotIn("seeds", o.ADAPTERS[PAGED])
        self.assertEqual(o.ADAPTERS[GMM]["seeds"], [0])
        self.assertIn("sum(group_sizes) == M", o.ADAPTERS[GMM]["seeds_reason"])


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class HumanTarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, rows, cls.p = run_oracle([_human("human_add", "t", KERNELS["outside_add"]), _human("model_add", "t", KERNELS["outside_add"], False)])
        cls.rows = {r["id"]: r for r in rows}

    def test_a_human_target_with_outside_work_gets_a_speed_verdict_and_records_the_outside_ops(self):
        r = self.rows["human_add"]
        self.assertEqual((r["verdict"], r["human_target"], r["is_kernel"]), ("pass", True, False), (r, self.p.stderr[-1500:]))
        self.assertIn("add", r["outside"])
        self.assertEqual(r["outside_disallowed"], ["add"])
        self.assertIn("heavy work outside pallas_call: add", r["is_kernel_reason"])
        self.assertTrue(r["correct"] and len(r["per_seed"]) == 3)

    def test_the_same_code_without_human_target_is_still_not_a_kernel(self):
        r = self.rows["model_add"]
        self.assertEqual((r["verdict"], r["is_kernel"]), ("not_a_kernel", False), r)
        self.assertNotIn("human_target", r)
        self.assertNotIn("per_seed", r)

    def test_a_declared_adapter_maps_the_inputs_for_a_human_target_only(self):
        global DRIVER
        old = DRIVER
        DRIVER = DRIVER.replace("o.main(sys.argv[2:])", TOY_ADAPTER + "o.main(sys.argv[2:])")
        try:
            _, rows, p = run_oracle([_human("h", "t", SWAPPED), _human("m", "t", SWAPPED, False)])
        finally:
            DRIVER = old
        rows = {r["id"]: r for r in rows}
        self.assertEqual((rows["h"]["verdict"], rows["h"]["adapter"]), ("pass", "b as a (1, 128) row, then x"), (rows, p.stderr[-1500:]))
        self.assertTrue(all(s["correct"] for s in rows["h"]["per_seed"]))
        self.assertNotEqual(rows["m"]["verdict"], "pass", rows["m"])
        self.assertNotIn("adapter", rows["m"])


REF_PAGED_FULL, REF_GMM_FULL = ((LEAK / t / "baseline.py").read_text() for t in (PAGED, GMM))


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class RealAdapters(unittest.TestCase):
    """6p and 11p: the real level 2 human kernels against the real level 1 references, traced at full size (--static-only: nothing
    executed), then run at toy size in interpret mode."""

    def test_full_size_trace_fails_without_the_adapter_and_passes_with_it(self):
        refs = {PAGED: REF_PAGED_FULL, GMM: REF_GMM_FULL}
        hs = [_human("h6", PAGED, _level2(PAGED)), _human("h11", GMM, _level2(GMM)),
              _human("m6", PAGED, _level2(PAGED), False), _human("m11", GMM, _level2(GMM), False)]
        _, rows, p = run_oracle(hs, "--static-only", refs=refs)
        rows = {r["id"]: r for r in rows}
        for i in ("h6", "h11"):
            r = rows[i]
            self.assertEqual((r["verdict"], r["human_target"]), ("static_ok", True), (r, p.stderr[-2000:]))
            self.assertTrue(r["adapter"] and r["pallas_calls"] >= 1 and r["output_depends_on_pallas"], r)
            self.assertIn("is_kernel_reason", r)
        self.assertEqual(rows["h11"]["static_argnums"], [3])  # the level 1 reference keeps its static max_expert_size
        # R2: gmm's make_group_metadata is small values computed only from group_sizes (int): cumsum, compares, scatter, and
        # jnp.histogram's float32 bin edges and searchsorted on the int tile ids. Metadata, so 11p is a kernel.
        self.assertEqual((rows["h11"]["is_kernel"], rows["h11"]["outside_disallowed"]), (True, []), rows["h11"])
        self.assertTrue({"cumsum", "gather", "lt_to"} <= set(rows["h11"]["outside_metadata_ok"]), rows["h11"])
        self.assertIn("only accepts 3 positional arguments", rows["m11"]["msg"])  # the first real run's failures
        self.assertIn("takes 5 positional arguments but 6 were given", rows["m6"]["msg"])

    def test_adapted_inputs_are_level_2s_own_inputs_and_the_outputs_match_level_1_in_shape(self):
        v = _probe(ADAPTER_PROBE, REF_PAGED_FULL, _level2(PAGED), REF_GMM_FULL, _level2(GMM))
        self.assertEqual(v, {"6p": {"inputs": True, "shape": True, "dtypes": ["bfloat16", "bfloat16"]},
                             "11p": {"inputs": True, "shape": True, "dtypes": ["float32", "bfloat16"]}})  # gmm accumulates to float32

    def test_paged_human_kernel_is_correct_on_every_seed_at_toy_size(self):
        _, rows, p = run_oracle([_human("h6", PAGED, _level2(PAGED, "TUNED_PARAMS = {'pages_per_compute_block': 2}"))],
                                refs={PAGED: REF_PAGED})
        r = rows[0]
        self.assertEqual(r["verdict"], "pass", (r, p.stderr[-2000:]))
        self.assertIn("head_dim**-0.5", r["adapter"])
        self.assertEqual([s["seed"] for s in r["per_seed"]], [0, 1, 2])  # no seed override: structure seeds included
        self.assertNotIn("seeds_used", r)

    def test_gmm_human_kernel_runs_seed_0_only_and_student_items_still_face_every_seed(self):
        """gmm's upstream contract is sum(group_sizes) == M, which only seed 0 (JAXBench as written) keeps: the human target runs
        there only, by its adapter's declared override. Model items of the same task, in the same run, still get every seed."""
        _, rows, p = run_oracle([_human("h11", GMM, _level2(GMM, "TUNED_PARAMS = {'tiling': [8, 128, 128]}")),
                                 {"id": "dense", "task": GMM, "code": GMM_KERNELS["dense"], "teacher": "x"},
                                 {"id": "grouped", "task": GMM, "code": GMM_KERNELS["grouped"], "teacher": "x"}], refs={GMM: REF_GMM})
        rows = {r["id"]: r for r in rows}
        r = rows["h11"]
        self.assertEqual((r["verdict"], r["static_argnums"], r["seeds"], r["seeds_used"]), ("pass", [3], [0, 1, 2], [0]), (r, p.stderr[-2000:]))
        self.assertIn("sum(group_sizes) == M", r["seed_override_reason"])
        self.assertEqual([s["seed"] for s in r["per_seed"]], [0])
        self.assertLess(r["per_seed"][0]["err_kernel"], 1e-5, r)
        for i, oks in (("dense", {0: True, 1: False, 2: False}), ("grouped", {0: True, 1: True, 2: True})):
            m = rows[i]
            self.assertEqual({s["seed"]: s["ok"] for s in m["per_seed"]}, oks, m)
            self.assertNotIn("seeds_used", m)
            self.assertNotIn("seed_override_reason", m)
        self.assertEqual(rows["dense"]["verdict"], "wrong")


ADAPTER_PROBE = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [sys.argv[1]]
    import tpu_oracle as o
    o.init(cpu_smoke=True)
    import jax
    out = {}
    sig = lambda t: [(tuple(x.shape), str(x.dtype)) for x in jax.tree_util.tree_leaves(t)]
    for name, task, l1, l2 in (("6p", "6p_Paged_Attention", *sys.argv[2:4]), ("11p", "11p_Megablox_GMM", *sys.argv[4:6])):
        t = o.Task(task, l1)
        t.load()  # abstract level 1 inputs at full size (11p's static max_expert_size kept concrete): nothing built
        l2m = o.tc.load(l2, "l2_" + name, t.dir.name)
        adapted = jax.eval_shape(o.ADAPTERS[task]["fn"], tuple(t.abstract))
        want = t.jit(t.ref.workload).eval_shape(*t.abstract)
        got = jax.eval_shape(l2m.workload, *adapted)
        out[name] = {"inputs": sig(adapted) == sig(jax.eval_shape(l2m.create_inputs)), "shape": got.shape == want.shape,
                     "dtypes": [str(got.dtype), str(want.dtype)]}
    print("JSON" + json.dumps(out))
''')


# ---- R1/R2 (batch 3): a float gather outside is heavy work; small integer metadata outside is not ----

REF_PG = '''import jax, jax.numpy as jnp
def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    return jax.random.normal(k1, (16, 8, 128), dtype), jax.random.permutation(k2, 16)[:4].astype(jnp.int32)
def workload(kv_pages, page_indices):
    return kv_pages[page_indices].reshape(32, 128) * 2.0
'''
REF_GS = '''import jax, jax.numpy as jnp
def create_inputs(dtype=jnp.float32):
    return jax.random.normal(jax.random.key(0), (64, 128), dtype), jnp.full((8,), 8, jnp.int32)
def workload(x, group_sizes):
    gid = jnp.sum(jnp.arange(64)[:, None] >= jnp.cumsum(group_sizes)[None, :], axis=1)
    return x * (gid[:, None] + 1).astype(x.dtype)
'''
DOUBLE = '''
def _double(kv):
    def k(x_ref, o_ref):
        o_ref[...] = x_ref[...] * 2.0
    return pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct(kv.shape, kv.dtype), grid=(4,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))], out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)))(kv)
'''
R2_KERNELS = {
    # the gather of whole float pages runs in XLA, outside the kernel: the kernel only doubles
    "paged_gather_outside": HDR + DOUBLE + "def workload(kv_pages, page_indices):\n    return _double(kv_pages[page_indices].reshape(32, 128))\n",
    # the same gather spelled as a Python loop of lax.dynamic_slice at page_indices[j] (Lebron James 6711257577): every op is allowed
    # by name, so only the non-literal start gives it away
    "paged_dynamic_slice_outside": HDR + DOUBLE + ("def workload(kv_pages, page_indices):\n"
        "    pages = [jax.lax.dynamic_slice(kv_pages, (page_indices[j], 0, 0), (1, 8, 128)) for j in range(4)]\n"
        "    return _double(jnp.concatenate(pages, axis=0).reshape(32, 128))\n"),
    # group ends as int32 metadata, computed outside and passed in, as Megablox's make_group_metadata does
    "cumsum_metadata": HDR + '''
def _k(ends_ref, x_ref, o_ref):
    i = pl.program_id(0)
    g = jnp.sum((i * 8 >= ends_ref[...]).astype(jnp.int32))
    o_ref[...] = x_ref[...] * (g + 1).astype(x_ref.dtype)
def workload(x, group_sizes):
    ends = jnp.cumsum(group_sizes).reshape(1, 8)
    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), grid=(8,),
        in_specs=[pl.BlockSpec((1, 8), lambda i: (0, 0)), pl.BlockSpec((8, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)))(ends, x)
''',
}
_GS_CALL = HDR + '''
def _k(ends_ref, h_ref, x_ref, o_ref):
    i = pl.program_id(0)
    g = jnp.sum((i * 8 >= ends_ref[...]).astype(jnp.int32)) + 0 * jnp.sum(h_ref[...])
    o_ref[...] = x_ref[...] * (g + 1).astype(x_ref.dtype)
def _call(ends, h, x):
    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), grid=(8,),
        in_specs=[pl.BlockSpec((1, 8), lambda i: (0, 0)), pl.BlockSpec((1, 8), lambda i: (0, 0)), pl.BlockSpec((8, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)))(ends.reshape(1, 8), h.reshape(1, 8), x)
'''
# jnp.histogram bins the int group sizes with float32 edges: untainted, so metadata (as in 11p's make_group_metadata)
R2_KERNELS["histogram_metadata"] = _GS_CALL + ("def workload(x, group_sizes):\n"
                                               "    h = jnp.histogram(group_sizes, bins=8, range=(0, 15))[0].astype(jnp.int32)\n"
                                               "    return _call(jnp.cumsum(group_sizes), h, x)\n")
# an int derived from the float input: tainted, so argmax and the index arithmetic are heavy work
R2_KERNELS["argmax_of_float_input"] = _GS_CALL + ("def workload(x, group_sizes):\n"
                                                  "    i = jnp.argmax(x[:, 0]).astype(jnp.int32)\n"
                                                  "    return _call(jnp.cumsum(group_sizes) + 0 * i, group_sizes, x)\n")
# a float op on the small (128-element) float input b, outside: tainted however small
KERNELS["small_float_input"] = HDR + CALL + ("def _k(x_ref, b_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2.0 + b_ref[...] * 0.5\n"
                                             "def workload(x, b):\n    return _call(_k, x, b * 2.0)\n")


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class OutsideMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        its = [{"id": "pg", "task": "pg", "code": R2_KERNELS["paged_gather_outside"], "teacher": "x"},
               {"id": "pgds", "task": "pg", "code": R2_KERNELS["paged_dynamic_slice_outside"], "teacher": "x"},
               {"id": "gs", "task": "gs", "code": R2_KERNELS["cumsum_metadata"], "teacher": "x"},
               {"id": "bias", "task": "t", "code": KERNELS["outside_add"], "teacher": "x"},
               {"id": "hist", "task": "gs", "code": R2_KERNELS["histogram_metadata"], "teacher": "x"},
               {"id": "argmax", "task": "gs", "code": R2_KERNELS["argmax_of_float_input"], "teacher": "x"},
               {"id": "small", "task": "t", "code": KERNELS["small_float_input"], "teacher": "x"}]
        _, rows, cls.p = run_oracle(its, refs={"pg": REF_PG, "gs": REF_GS, "t": REF})
        cls.rows = {r["id"]: r for r in rows}

    def test_a_float_page_gather_outside_the_kernel_is_not_a_kernel(self):
        r = self.rows["pg"]
        self.assertEqual((r["verdict"], r["is_kernel"]), ("not_a_kernel", False), (r, self.p.stderr[-1500:]))
        self.assertIn("gather", r["outside_disallowed"])
        self.assertIn("gather", r["feedback"])

    def test_the_same_gather_as_a_dynamic_slice_loop_is_not_a_kernel(self):
        r = self.rows["pgds"]
        self.assertEqual((r["verdict"], r["is_kernel"]), ("not_a_kernel", False), (r, self.p.stderr[-1500:]))
        self.assertIn("dynamic_slice(dynamic start)", r["outside_disallowed"])

    def test_int32_cumsum_metadata_outside_is_allowed_and_the_kernel_runs(self):
        r = self.rows["gs"]
        self.assertTrue(r["is_kernel"], (r, self.p.stderr[-1500:]))
        self.assertEqual(r["outside_disallowed"], [])
        self.assertIn("cumsum", r["outside_metadata_ok"])
        self.assertEqual(r["verdict"], "pass", r)

    def test_an_f32_bias_epilogue_outside_still_fails(self):
        r = self.rows["bias"]
        self.assertEqual((r["verdict"], r["is_kernel"], r["outside_disallowed"], r["outside_metadata_ok"]),
                         ("not_a_kernel", False, ["add"], []), r)

    def test_histogram_of_small_int_metadata_passes(self):
        r = self.rows["hist"]
        self.assertEqual((r["verdict"], r["is_kernel"], r["outside_disallowed"]), ("pass", True, []), (r, self.p.stderr[-1500:]))
        self.assertIn("cumsum", r["outside_metadata_ok"])

    def test_an_int_derived_from_a_float_input_is_tainted(self):
        r = self.rows["argmax"]
        self.assertEqual((r["verdict"], r["is_kernel"]), ("not_a_kernel", False), r)
        self.assertIn("argmax", r["outside_disallowed"])
        self.assertIn("cumsum", r["outside_metadata_ok"])  # the group sizes' own cumsum is still metadata

    def test_a_float_op_on_a_small_float_input_still_fails(self):
        r = self.rows["small"]
        self.assertEqual((r["verdict"], r["outside_disallowed"], r["outside_metadata_ok"]), ("not_a_kernel", ["mul"], []), r)


# ---- numpy-seeded structure (Lebron James 6710707491, item 2a): 51p/54p/58p build index structure with np.random.default_rng(42) ----

REF_NP = '''import jax, jax.numpy as jnp, numpy as np
def create_inputs(dtype=jnp.float32):
    perm = jnp.asarray(np.random.default_rng(42).permutation(16), jnp.int32)
    return jax.random.normal(jax.random.key(0), (16, 128), dtype), perm
def workload(x, perm):
    return x[perm] * 2.0
'''
NP_PROBE = textwrap.dedent('''
    import json, pathlib, sys, tempfile
    sys.path[:0] = [sys.argv[1]]
    import tpu_oracle as o
    o.init(cpu_smoke=True)
    import numpy as np
    orig = np.random.default_rng
    ref = o.tc.load(sys.argv[2], "ref", tempfile.mkdtemp())
    perms = [[int(v) for v in o.make_inputs(ref, s)[1]] for s in (0, 1, 2)]
    print("JSON" + json.dumps({"perms": perms, "plain": [int(v) for v in orig(42).permutation(16)], "restored": np.random.default_rng is orig,
                               "vary": o.int_inputs_vary("t", ref, [0, 1, 2])[0]}))
''')


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class NumpySeeds(unittest.TestCase):
    def test_seed_0_is_numpy_as_written_and_seeds_1_and_2_draw_new_permutations(self):
        v = _probe(NP_PROBE, REF_NP)
        p0, p1, p2 = v["perms"]
        self.assertEqual(p0, v["plain"])
        self.assertTrue(p0 != p1 and p0 != p2 and p1 != p2, v)
        self.assertEqual(sorted(p1), list(range(16)))  # still a permutation: only the draw changes
        self.assertTrue(v["restored"] and v["vary"], v)

    def test_an_unseeded_create_inputs_is_refused_as_a_harness_fault(self):
        # Lebron James 6711257577: default_rng() is fresh entropy, so the reference and the kernel would see different data
        unseeded = REF_NP.replace("np.random.default_rng(42)", "np.random.default_rng()")
        _, rows, p = run_oracle([{"id": "ref/t", "task": "t", "code": ""}], "--reference-only", refs={"t": unseeded})
        self.assertNotEqual(rows[0]["verdict"], "reference_ok", (rows, p.stderr[-1500:]))
        self.assertIn("not deterministic", rows[0]["feedback"])

    def test_rows_say_the_int_inputs_vary(self):
        _, rows, p = run_oracle([{"id": "ref/t", "task": "t", "code": ""}], "--reference-only", refs={"t": REF_NP})
        self.assertEqual((rows[0]["verdict"], rows[0]["int_inputs_vary"]), ("reference_ok", True), (rows, p.stderr[-1500:]))
