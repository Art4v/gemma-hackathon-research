"""oracle/keys.py: a chunked answer key must equal the reference's own maths (checked at a size where both fit)."""
import importlib.util, json, os, pathlib, subprocess, sys, textwrap, unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tests"))
from test_tpu_oracle import PY  # noqa: E402

PROBE = textwrap.dedent('''
    import importlib.util, json, sys
    sys.path[:0] = [sys.argv[1]]
    import jax, jax.numpy as jnp, numpy as np, keys
    s = importlib.util.spec_from_file_location("ref", sys.argv[2]); ref = importlib.util.module_from_spec(s); s.loader.exec_module(ref)
    out = {}
    for (B, S, Hq, Hkv, D) in ((2, 64, 8, 2, 16), (1, 128, 16, 4, 32), (3, 32, 4, 4, 8)):
        k1, k2, k3 = jax.random.split(jax.random.key(B * 1000 + S), 3)
        q = jax.random.normal(k1, (B, S, Hq, D), jnp.float32)
        k = jax.random.normal(k2, (B, S, Hkv, D), jnp.float32)
        v = jax.random.normal(k3, (B, S, Hkv, D), jnp.float32)
        with jax.default_matmul_precision("highest"):
            want, got = np.asarray(ref.workload(q, k, v)), np.asarray(jax.jit(keys.gqa_chunked)(q, k, v))
        out[f"{B}x{S}x{Hq}x{Hkv}x{D}"] = {"shape": list(got.shape) == list(want.shape), "max_abs": float(np.abs(got - want).max())}
    print("JSON" + json.dumps(out))
''')


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class ChunkedGqaKey(unittest.TestCase):
    def test_chunked_key_equals_the_2p_reference_at_small_sizes(self):
        ref = HERE.parent / "benchmarks/jaxbench/level1/2p_GQA_Attention/baseline.py"
        p = subprocess.run([PY, "-c", PROBE, str(HERE), str(ref)], capture_output=True, text=True, timeout=600,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
        line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
        self.assertTrue(line, p.stderr[-2000:])
        for size, r in json.loads(line[0][4:]).items():
            self.assertTrue(r["shape"], size)
            self.assertLess(r["max_abs"], 1e-5, (size, r))  # same maths in f32 at highest precision: only summation order differs

    def test_the_key_registry_names_only_2p(self):
        import keys  # noqa: F401  (oracle/ is on sys.path via test_tpu_oracle)
        self.assertEqual(set(keys.KEYS), {"2p_GQA_Attention"})


if __name__ == "__main__":
    unittest.main()


KEY_PROBE = textwrap.dedent('''
    import json, sys
    sys.path[:0] = [sys.argv[1]]
    import tpu_oracle as o, keys
    o.init(cpu_smoke=True)
    import jax.numpy as jnp, numpy as np
    ref = "import jax, jax.numpy as jnp\\ndef create_inputs(dtype=jnp.float32):\\n    return (jnp.arange(8, dtype=dtype),)\\ndef workload(x):\\n    return x * 2\\n"
    seen = []
    keys.KEYS["t_chunked"] = lambda x: (seen.append(str(x.dtype)), x * 2 + 1)[1]  # distinguishable from the reference
    out = {}
    for name in ("t_chunked", "t_plain"):
        task = o.Task(name, ref); task.load()
        st = {"precision": "f32_highest", "failed": []}
        want = task._key(task.inputs(0), st)
        out[name] = {"key": np.asarray(want).tolist()[:3], "state": st}
    out["seen"] = seen
    print("JSON" + json.dumps(out))
''')


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class KeyWiring(unittest.TestCase):
    def test_f32_key_uses_the_registered_chunked_function_and_says_so(self):
        p = subprocess.run([PY, "-c", KEY_PROBE, str(HERE)], capture_output=True, text=True, timeout=600,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"})
        line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
        self.assertTrue(line, p.stderr[-2000:])
        v = json.loads(line[0][4:])
        self.assertEqual(v["t_chunked"]["key"], [1.0, 3.0, 5.0])  # the chunked function ran, on float32 inputs
        self.assertEqual(v["seen"], ["float32"])
        self.assertTrue(v["t_chunked"]["state"].get("chunked_key"))
        self.assertEqual(v["t_plain"]["key"], [0.0, 2.0, 4.0])
        self.assertFalse(v["t_plain"]["state"].get("chunked_key"))

    def test_keys_py_ships_in_packs_and_moves_the_oracle_sha(self):
        sys.path.insert(0, str(HERE))
        import make_pack, tpu_oracle  # noqa: E401
        self.assertIn("keys.py", make_pack.ORACLE_FILES)
        self.assertIn("keys.py", (HERE / "tpu_oracle.py").read_text().split("def _provenance", 1)[1].split("def ", 1)[0])
