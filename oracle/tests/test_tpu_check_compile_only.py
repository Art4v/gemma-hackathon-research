"""paper/tpu/tpu_check.py --compile-only (H3 gate vi), run on CPU with hand-written kernels. On CPU the compile is interpret mode, so
this tests the plumbing and the row, not Mosaic. The reference is 16 GiB at full size: allocating it would fail, compiling must not."""
import importlib.util, json, os, pathlib, subprocess, sys, tempfile, unittest

HERE = pathlib.Path(__file__).resolve().parent.parent
PY = os.environ.get("SERA_JAX_PY") or (sys.executable if importlib.util.find_spec("jax") else "")

REF = '''import jax, jax.numpy as jnp
def cap_shape(shape, cap):
    return tuple(min(d, cap) for d in shape) if cap else tuple(shape)
def create_inputs(dtype=jnp.float32, cap=None):
    return (jax.random.uniform(jax.random.key(0), cap_shape((65536, 65536), cap), dtype),)
def workload(x):
    return x * 2.0
'''
GOOD = '''import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
def _k(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0
def workload(x):
    m, n = x.shape
    bm, bn = min(512, m), min(1024, n)
    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), grid=(m // bm, n // bn),
        in_specs=[pl.BlockSpec((bm, bn), lambda i, j: (i, j))], out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)))(x)
'''
BROKEN = "def workload(x):\n    raise ValueError('no kernel here')\n"


def run(items, *flags):
    with tempfile.TemporaryDirectory() as d:
        pack = pathlib.Path(d, "pack.json")
        pack.write_text(json.dumps({"dev_refs": {"t": REF}, "train_refs": {}, "dev": items, "train": []}))
        p = subprocess.run([PY, str(HERE / "tpu_check.py"), str(pack), str(pathlib.Path(d, "out.jsonl")), "--which", "dev", *flags],
                           capture_output=True, text=True, timeout=600, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        out = pathlib.Path(d, "out.jsonl")
        assert out.exists(), p.stderr[-800:]
        rows = [json.loads(l) for l in out.read_text().splitlines()]
    return rows[0], rows[1:]


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class CompileOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta, rows = run([{"id": "good", "task": "t", "code": GOOD, "teacher": "x"}, {"id": "broken", "task": "t", "code": BROKEN}],
                             "--cpu-smoke", "--compile-only")
        cls.rows = {r["id"]: r for r in rows}

    def test_meta_says_compile_only(self):
        self.assertTrue(self.meta["meta"] and self.meta["compile_only"])

    def test_every_row_has_the_compile_fields(self):
        for r in self.rows.values():
            for k in ("id", "task", "stack", "chip", "mode", "compiled", "full_input_bytes", "shapes"):
                self.assertIn(k, r, (k, r))
            self.assertEqual(r["mode"], "compile_only")
            self.assertEqual(r["shapes"], [[65536, 65536]])               # full size, from eval_shape
            self.assertEqual(r["full_input_bytes"], 65536 * 65536 * 4)
            self.assertNotIn("t_kernel_s", r)                             # nothing is timed
            self.assertNotIn("correct", r)                                # nothing is run

    def test_good_kernel_compiles_or_reports_why(self):
        r = self.rows["good"]
        self.assertTrue(r["compiled"] or (r.get("error") and r.get("failed_stage")), r)
        self.assertEqual(r["teacher"], "x")
        if r["compiled"]:
            self.assertEqual(r["pallas_calls"], 1)

    def test_failing_kernel_is_a_row_with_error_type_and_message(self):
        r = self.rows["broken"]
        self.assertEqual((r["compiled"], r["failed_stage"], r["error"]), (False, "kernel", "ValueError"))
        self.assertIn("no kernel here", r["msg"])

    def test_default_cpu_smoke_path_is_unchanged(self):
        meta, rows = run([{"id": "good", "task": "t", "code": GOOD}], "--cpu-smoke")
        self.assertNotIn("compile_only", meta)
        self.assertEqual((rows[0]["ran"], rows[0]["correct"], rows[0]["shapes"]), (True, True, [[256, 256]]))
        self.assertNotIn("mode", rows[0])


if __name__ == "__main__":
    unittest.main()
