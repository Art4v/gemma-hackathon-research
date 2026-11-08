"""paper/tpu/tpu_check.py was refactored so oracle/tpu_oracle.py can import it. 1,031 committed v6e labels came from its CLI, so the CLI's
output must not change: the refactored script and the version at LABELS_COMMIT run the same packs and must write the same bytes, to OUT
and to stdout, in every CPU-runnable mode (default, --shrink, --compile-only, --shard/--limit, resume). Timing runs only on a TPU; that
code path was not edited. Also: importing the module has no CLI side effects."""
import importlib.util, json, os, pathlib, subprocess, sys, tempfile, unittest

HERE = pathlib.Path(__file__).resolve().parent.parent
PY = os.environ.get("SERA_JAX_PY") or (sys.executable if importlib.util.find_spec("jax") else "")
LABELS_COMMIT = "9e32ef4"  # last commit to touch tpu_check.py before the oracle refactor

REF = '''import jax, jax.numpy as jnp
def cap_shape(shape, cap):
    return tuple(min(d, cap) for d in shape) if cap else tuple(shape)
def create_inputs(dtype=jnp.float32, cap=None):
    return (jax.random.uniform(jax.random.key(0), cap_shape((512, 1024), cap), dtype),)
def workload(x):
    return x * 2.0
'''
REF_NOCAP = "import jax, jax.numpy as jnp\ndef create_inputs(dtype):\n    return (jax.random.normal(jax.random.key(1), (16, 384), dtype),)\n" \
            "def workload(x):\n    return x + 1.0\n"
HDR = "import jax, jax.numpy as jnp\nfrom jax.experimental import pallas as pl\n"
TILED = HDR + '''def _k(x_ref, o_ref):
    o_ref[...] = x_ref[...] * {f}
def workload(x):
    m, n = x.shape
    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), grid=(m // 8, n // 128),
        in_specs=[pl.BlockSpec((8, 128), lambda i, j: (i, j))], out_specs=pl.BlockSpec((8, 128), lambda i, j: (i, j)))(x)
'''
ITEMS = [{"id": "good", "task": "t", "code": TILED.format(f="2.0"), "teacher": "x"},
         {"id": "wrong", "task": "t", "code": TILED.format(f="2.5"), "model": "m", "arm": "a"},
         {"id": "plain", "task": "t", "code": "import jax.numpy as jnp\ndef workload(x):\n    return x * 2.0\n"},
         {"id": "broken", "task": "t", "code": "def workload(x):\n    raise ValueError('no kernel here')\n"},
         {"id": "nocap", "task": "u", "code": "import jax.numpy as jnp\ndef workload(x):\n    return x + 1.0\n"}]
MODES = [["--cpu-smoke"], ["--cpu-smoke", "--shrink", "2"], ["--cpu-smoke", "--compile-only"], ["--cpu-smoke", "--shard", "1/2", "--limit", "2"],
         ["--cpu-smoke", "--which", "train", "--stack", "other"]]


def old_script():
    p = subprocess.run(["git", "-C", str(HERE), "show", f"{LABELS_COMMIT}:./paper/tpu/tpu_check.py"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def run(script, flags, prior=None):
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "tpu_check.py").write_text(script)
        (d / "pack.json").write_text(json.dumps({"dev_refs": {"t": REF, "u": REF_NOCAP}, "train_refs": {"t": REF, "u": REF_NOCAP},
                                                  "dev": ITEMS, "train": ITEMS[:2]}))
        if prior is not None:
            (d / "out.jsonl").write_bytes(prior)
        # run from inside the temp dir with relative paths, so no absolute path can differ between the two runs
        p = subprocess.run([PY, "tpu_check.py", "pack.json", "out.jsonl", *flags], cwd=d, capture_output=True, text=True, timeout=600,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
        return (d / "out.jsonl").read_bytes() if (d / "out.jsonl").exists() else b"", p.stdout, p.returncode


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class ByteCompatible(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = old_script()
        cls.new = (HERE / "tpu_check.py").read_text()

    def setUp(self):
        if self.old is None:
            self.skipTest(f"git show {LABELS_COMMIT} unavailable")

    def test_same_bytes_in_every_cpu_mode(self):
        for flags in MODES:
            old, new = run(self.old, flags), run(self.new, flags)
            self.assertTrue(old[0], (flags, "old version wrote nothing"))
            self.assertEqual(old, new, flags)

    def test_resume_appends_the_same_bytes(self):
        prior = (json.dumps({"meta": True}) + "\n" + json.dumps({"id": "good", "task": "t"}) + "\n").encode()
        old, new = run(self.old, ["--cpu-smoke"], prior), run(self.new, ["--cpu-smoke"], prior)
        self.assertEqual(old, new)
        self.assertEqual(old[0].count(b'"id": "good"'), 1)


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class Importable(unittest.TestCase):
    def test_import_parses_no_argv_and_patches_nothing(self):
        code = ("import sys; sys.argv = ['x', '--bogus']; sys.path.insert(0, %r)\n"
                "from jax.experimental import pallas as pl; orig = pl.pallas_call\n"
                "import signal, tpu_check as t\n"
                "assert t.a is None and pl.pallas_call is orig and signal.getsignal(signal.SIGALRM) is not t._alarm\n"
                "assert t.HBM_BUDGET_BYTES == 28e9 and (t.WARMUP, t.REPS) == (3, 20)\n"
                "pack = {'dev_refs': {'t': 'R'}, 'train_refs': {}, 'dev': [{'id': str(k), 'task': 't'} for k in range(5)], 'train': []}\n"
                "refs, items = t.select(pack, 'dev', '1/2', 0)\n"
                "assert refs == {'t': 'R'} and [i['id'] for i in items] == ['1', '3']\n"
                "print('OK')\n") % str(HERE)
        p = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        self.assertIn("OK", p.stdout, p.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
