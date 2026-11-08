"""oracle/tpu_oracle.py: the verdict on a (task, solution). The pure parts (error classes, feedback, the speed bar, the kernel check on
a jaxpr) are tested without JAX on fake jaxprs and injected timings. The JAX tests run the oracle on CPU in Pallas interpret mode
through a driver that replaces the timer, on small hand-written kernels only."""
import importlib.util, json, os, pathlib, subprocess, sys, tempfile, textwrap, unittest

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
        c = V()
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


# Real chip messages, from earlier v6e runs.
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
    """For vmem and compile failures the feedback gives the numbers before any trimmed part, with the HLO attribute noise gone."""

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
            self.assertLessEqual(len(fb), len(o.LABELS[cls]) + 2 + len(o.trim("x" * 10 ** 4)), fb)
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


class XlaEq(unittest.TestCase):
    """The solution's error may be at most twice XLA default's own, floored at 1e-6. One-sided."""

    def test_kernel_more_accurate_than_xla_passes(self):
        self.assertTrue(o.xla_eq(0.03, 0.05))
        rows = seed_rows(True, True, True)
        ek, ex = 0.03, 0.05
        rows[1].update(correct=False, err_kernel=ek, err_xla_default=ex, xla_eq=o.xla_eq(ek, ex), ok=o.xla_eq(ek, ex))
        self.assertEqual(o.final_verdict(rows, t_kernel=1.0, t_xla=1.0, speed_bar=o.SPEED_BAR)[0], "pass")

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


class StaticArgnumsParse(unittest.TestCase):
    def test_reads_the_harness_jit_of_workload(self):
        self.assertEqual(o.static_argnums("fn = jax.jit(workload, static_argnums=(3,))"), (3,))
        self.assertEqual(o.static_argnums("fn = jax.jit(workload, static_argnums=2)"), (2,))
        self.assertEqual(o.static_argnums("fn = jax.jit( workload,\n  static_argnums=[1, 4])"), (1, 4))
        self.assertEqual(o.static_argnums("fn = jax.jit(workload)\ng = jax.jit(other, static_argnums=(1,))"), ())
        self.assertEqual(o.static_argnums(REF), ())


class OnlyTheJob(unittest.TestCase):
    """The oracle does the four steps and nothing else (oracle/DESCRIPTION.md): the removed modes are gone from the CLI."""

    def test_defaults(self):
        a = o.parse_args(["p.json", "o.jsonl"])
        self.assertEqual((a.seeds, a.speed_bar, a.timeout, a.shard), (3, 0.7, 480, "0/1"))
        self.assertEqual(o.ORACLE_FILES, ("tpu_oracle.py", "tpu_check.py", "outside_ops.py"))

    def test_removed_modes_are_refused(self):
        for flag in ("--reference-only", "--sensitivity", "--input-dtype=float32", "--which=train"):
            with self.subTest(flag), self.assertRaises(SystemExit):
                o.parse_args(["p.json", "o.jsonl", flag])


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
    # correct, but the add is outside the kernel
    "outside_add": HDR + CALL + "def _k(x_ref, b_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2.0\n"
                                "def workload(x, b):\n    return _call(_k, x, b) + b\n",
    # a float op on the small (128-element) float input b, outside: tainted however small
    "small_float_input": HDR + CALL + "def _k(x_ref, b_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2.0 + b_ref[...] * 0.5\n"
                                      "def workload(x, b):\n    return _call(_k, x, b * 2.0)\n",
}

DRIVER = '''import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import tpu_oracle as o
o.init(cpu_smoke=True)
import jax
b0 = jax.random.normal(jax.random.split(jax.random.key(0))[1], (128,), jax.numpy.float32)
pack = json.loads(pathlib.Path(sys.argv[2]).read_text())
for it in pack["items"]:
    it["code"] = it["code"].replace("{c!r}", repr(float(b0[0])))
pathlib.Path(sys.argv[2]).write_text(json.dumps(pack))
o.tc.timed_stats = lambda f, inputs: (1.0, 0.9, 1.2)  # the timer is mocked: (median, p10, p90), so speed_vs_xla is 1.0
o.main(sys.argv[2:])
'''


def run_oracle(items, *flags, refs=None, out_rows=None, oracle_dir=HERE, env=None):
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "pack.json").write_text(json.dumps({"refs": refs or {"t": REF}, "items": items}))
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


def _probe(script, *args):
    p = subprocess.run([PY, "-c", script, str(HERE), *args], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "JAX_PLATFORMS": "cpu"})
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    assert line, p.stderr[-2000:]
    return json.loads(line[0][4:])


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class TimedStats(unittest.TestCase):
    def test_median_p10_p90_are_ordered_over_warmup_plus_reps_calls(self):
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
        v = _probe(script)
        med, p10, p90 = v["stats"]
        self.assertTrue(0 < p10 <= med <= p90, v)
        self.assertLess(p10, p90, v)  # sleeps of 0..4 ms: the spread is real
        self.assertEqual(v["calls"], v["want"])


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metas, rows, cls.proc = run_oracle(items("plain_jnp", "good", "seed0_only", "dropped", "raises"))
        cls.rows = {r["id"]: r for r in rows}

    def test_every_kernel_gets_one_row_with_the_provenance_fields(self):
        self.assertEqual(set(self.rows), {"plain_jnp", "good", "seed0_only", "dropped", "raises"}, self.proc.stderr[-800:])
        for r in self.rows.values():
            for k in ("id", "task", "verdict", "feedback", "seeds", "device_kind", "jax", "oracle_commit", "oracle_sha", "speed_bar", "teacher"):
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
        self.assertEqual(set(r["per_seed"][0]), {"seed", "correct", "err_kernel", "err_xla_default", "xla_eq", "ok"})
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


# x @ w done outside the kernel and copied through a tiled kernel, spelled five ways a name ban would miss; and the honest kernel
REF_MM = ("import jax, jax.numpy as jnp\n"
          "def create_inputs(dtype=jnp.float32):\n    k = jax.random.key(0)\n"
          "    return jax.random.normal(k, (64, 128), dtype), jax.random.normal(k, (128, 128), dtype)\n"
          "def workload(x, w):\n    return x @ w\n")
MM_HDR = "import jax, jax.numpy as jnp\nfrom jax import lax\nfrom jax.experimental import pallas as pl\n"
COPY = ("def _copy(y):\n    def k(y_ref, o_ref):\n        o_ref[...] = y_ref[...]\n"
        "    return pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct(y.shape, y.dtype), grid=(y.shape[0] // 8,),\n"
        "        in_specs=[pl.BlockSpec((8, y.shape[1]), lambda i: (i, 0))], out_specs=pl.BlockSpec((8, y.shape[1]), lambda i: (i, 0)))(y)\n")
CHEATS = {
    "alias": "d = lax.dot_general\ndef workload(x, w):\n    return _copy(d(x, w, (((1,), (0,)), ((), ()))))\n",
    "getattr": "def workload(x, w):\n    return _copy(getattr(jnp, 'mat' + 'mul')(x, w))\n",
    "product_then_sum": "def workload(x, w):\n    p = x[:, :, None] * w[None]\n    return _copy(p.sum(axis=1))\n",
    "reduce_sum": "def workload(x, w):\n    return _copy(lax.reduce_sum(x[:, :, None] * w[None], axes=(1,)))\n",
    "add_reduce": "def workload(x, w):\n    return _copy(jnp.add.reduce(x[:, :, None] * w[None], axis=1))\n",
}
HONEST_MM = ("def workload(x, w):\n    def k(x_ref, w_ref, o_ref):\n        o_ref[...] = jnp.dot(x_ref[...], w_ref[...])\n"
             "    return pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct((x.shape[0], w.shape[1]), x.dtype), grid=(x.shape[0] // 8,),\n"
             "        in_specs=[pl.BlockSpec((8, x.shape[1]), lambda i: (i, 0)), pl.BlockSpec(w.shape, lambda i: (0, 0))],\n"
             "        out_specs=pl.BlockSpec((8, w.shape[1]), lambda i: (i, 0)))(x, w)\n")


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class CheatsAreNotKernels(unittest.TestCase):
    def test_every_cheat_is_caught_and_the_honest_kernel_passes(self):
        its = [{"id": n, "task": "mm", "code": MM_HDR + COPY + c} for n, c in CHEATS.items()]
        its.append({"id": "honest", "task": "mm", "code": MM_HDR + HONEST_MM})
        _, rows, p = run_oracle(its, refs={"mm": REF_MM})
        rows = {r["id"]: r for r in rows}
        for n in CHEATS:
            self.assertEqual(rows[n]["verdict"], "not_a_kernel", (n, rows[n], p.stderr[-800:]))
            self.assertTrue(set(rows[n]["outside_disallowed"]) & {"dot_general", "mul", "reduce_sum"}, (n, rows[n]))
        self.assertEqual((rows["honest"]["verdict"], rows["honest"]["outside_disallowed"], rows["honest"]["pallas_calls"]),
                         ("pass", [], 1), (rows["honest"], p.stderr[-800:]))


# a float gather outside is heavy work; small integer metadata outside is not
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
R2_KERNELS = {
    # the gather of whole float pages runs in XLA, outside the kernel: the kernel only doubles
    "paged_gather_outside": HDR + DOUBLE + "def workload(kv_pages, page_indices):\n    return _double(kv_pages[page_indices].reshape(32, 128))\n",
    # the same gather spelled as a Python loop of lax.dynamic_slice at page_indices[j]: every op is allowed by name, so only the
    # non-literal start gives it away
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
    # jnp.histogram bins the int group sizes with float32 edges: untainted, so metadata
    "histogram_metadata": _GS_CALL + ("def workload(x, group_sizes):\n"
                                      "    h = jnp.histogram(group_sizes, bins=8, range=(0, 15))[0].astype(jnp.int32)\n"
                                      "    return _call(jnp.cumsum(group_sizes), h, x)\n"),
    # an int derived from the float input: tainted, so argmax and the index arithmetic are heavy work
    "argmax_of_float_input": _GS_CALL + ("def workload(x, group_sizes):\n"
                                         "    i = jnp.argmax(x[:, 0]).astype(jnp.int32)\n"
                                         "    return _call(jnp.cumsum(group_sizes) + 0 * i, group_sizes, x)\n"),
}


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class OutsideMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        its = [{"id": "pg", "task": "pg", "code": R2_KERNELS["paged_gather_outside"]},
               {"id": "pgds", "task": "pg", "code": R2_KERNELS["paged_dynamic_slice_outside"]},
               {"id": "gs", "task": "gs", "code": R2_KERNELS["cumsum_metadata"]},
               {"id": "bias", "task": "t", "code": KERNELS["outside_add"]},
               {"id": "hist", "task": "gs", "code": R2_KERNELS["histogram_metadata"]},
               {"id": "argmax", "task": "gs", "code": R2_KERNELS["argmax_of_float_input"]},
               {"id": "small", "task": "t", "code": KERNELS["small_float_input"]}]
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


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class NativeDtype(unittest.TestCase):
    def test_inputs_keep_the_tasks_own_default_dtype(self):
        probe = textwrap.dedent('''
            import json, sys, types
            sys.path.insert(0, sys.argv[1])
            import tpu_oracle as o
            o.init(cpu_smoke=True)
            m = types.ModuleType("ref")
            exec("import jax, jax.numpy as jnp\\ndef create_inputs(dtype=jnp.bfloat16):\\n"
                 "    return (jax.random.normal(jax.random.key(0), (8, 128), dtype),)\\n", m.__dict__)
            print("JSON" + json.dumps([str(o.make_inputs(m, s)[0].dtype) for s in (0, 1)]))
        ''')
        self.assertEqual(_probe(probe), ["bfloat16", "bfloat16"])


BOOM = '''class _Boom:
    def __enter__(self): raise RuntimeError("contract_precision<fp32> is not supported here")
    def __exit__(self, *a): return False
o.jax.default_matmul_precision = lambda *a, **k: _Boom()
'''


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class ReferenceRobustness(unittest.TestCase):
    """A reference whose 'highest'-precision run won't compile falls back to default precision and says so; a reference that defines
    a dataclass loads (its module is registered)."""

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
            print("JSON{}")
        ''')
        self.assertEqual(_probe(probe), {})


# a reference whose harness jits workload with static_argnums: input 2 is a Python int used in a shape, so it must stay concrete
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
# the static position is an argument the harness passes itself, beyond what create_inputs returns
REF_STATIC_BEYOND = REF.replace("def workload(x, b):", "def workload(x, b, k=None):") + \
    "def benchmark():\n    return jax.jit(workload, static_argnums=(2,))(*create_inputs(), 3)\n"


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class StaticArgs(unittest.TestCase):
    def test_kernel_for_a_reference_with_a_static_arg_passes_and_the_row_records_it(self):
        _, rows, p = run_oracle([{"id": "good_static", "task": "t", "code": GOOD_STATIC}], refs={"t": REF_STATIC})
        r = rows[0]
        self.assertEqual(r["verdict"], "pass", (r, p.stderr[-1500:]))
        self.assertEqual(r["static_argnums"], [2])
        self.assertTrue(all(s["correct"] for s in r["per_seed"]) and r["seeds_distinct"])
        self.assertEqual(r["shapes"], [[2048], [128]])

    def test_tasks_without_static_args_record_nothing(self):
        for ref in (REF, REF_STATIC_BEYOND):
            _, rows, p = run_oracle(items("good"), refs={"t": ref})
            self.assertEqual(rows[0]["verdict"], "pass", (rows, p.stderr[-1500:]))
            self.assertNotIn("static_argnums", rows[0])


REF_BF16 = REF.replace("def create_inputs(dtype=jnp.float32):", "def create_inputs(dtype=jnp.bfloat16):")


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class AnswerKey(unittest.TestCase):
    """The key is a real higher-precision answer. A bf16 reference is run on float32-cast inputs at 'highest' (f32_highest), so XLA's
    own bf16 output has a non-zero error against it and xla_eq keeps its meaning."""

    def test_bf16_reference_gets_an_f32_highest_key(self):
        _, rows, p = run_oracle(items("good"), refs={"t": REF_BF16})
        r = rows[0]
        self.assertEqual(r["ref_precision"], "f32_highest", (r, p.stderr[-1500:]))
        self.assertGreater(max(s["err_xla_default"] for s in r["per_seed"]), 0.0, r)


# numpy-seeded inputs: a create_inputs may draw with np.random.default_rng(int); seeds >= 1 fold the seed into it too
REF_NP = '''import jax, jax.numpy as jnp, numpy as np
def create_inputs(dtype=jnp.float32):
    perm = jnp.asarray(np.random.default_rng(42).permutation(16), jnp.int32)
    return jax.random.normal(jax.random.key(0), (16, 128), dtype), perm
def workload(x, perm):
    return x[perm] * 2.0
'''
NP_PROBE = textwrap.dedent('''
    import json, sys, tempfile
    sys.path[:0] = [sys.argv[1]]
    import tpu_oracle as o
    o.init(cpu_smoke=True)
    import numpy as np
    orig = np.random.default_rng
    ref = o.tc.load(sys.argv[2], "ref", tempfile.mkdtemp())
    perms = [[int(v) for v in o.make_inputs(ref, s)[1]] for s in (0, 1, 2)]
    print("JSON" + json.dumps({"perms": perms, "plain": [int(v) for v in orig(42).permutation(16)], "restored": np.random.default_rng is orig}))
''')
# an unseeded numpy draw: fresh entropy on every call, so the reference and the solution would see different data
REF_UNSEEDED = REF.replace("import jax, jax.numpy as jnp", "import jax, jax.numpy as jnp, numpy as np").replace(
    "jax.random.normal(k2, (128,), dtype)", "jnp.asarray(np.random.default_rng().standard_normal(128), dtype)")


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class NumpySeeds(unittest.TestCase):
    def test_seed_0_is_numpy_as_written_and_seeds_1_and_2_draw_new_permutations(self):
        v = _probe(NP_PROBE, REF_NP)
        p0, p1, p2 = v["perms"]
        self.assertEqual(p0, v["plain"])
        self.assertTrue(p0 != p1 and p0 != p2 and p1 != p2, v)
        self.assertEqual(sorted(p1), list(range(16)))  # still a permutation: only the draw changes
        self.assertTrue(v["restored"], v)

    def test_an_unseeded_create_inputs_is_a_harness_fault_not_the_solutions(self):
        _, rows, p = run_oracle(items("good"), refs={"t": REF_UNSEEDED})
        r = rows[0]
        self.assertEqual((r["verdict"], r["failed_stage"]), ("harness_oom", "ref"), (r, p.stderr[-1500:]))
        self.assertIn("not deterministic", r["feedback"])


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
