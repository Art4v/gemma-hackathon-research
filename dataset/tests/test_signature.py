"""dataset/scripts/signature.py: canonicalisation and collapse on fake jaxprs (no JAX); real tracing of exam references when a JAX
interpreter is available (SERA_JAX_PY)."""
import importlib.util, os, pathlib, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))
PY = os.environ.get("SERA_JAX_PY") or (sys.executable if importlib.util.find_spec("jax") else "")

import signature as sg  # noqa: E402


class P:
    def __init__(self, name): self.name = name


class Eqn:
    def __init__(self, name, params=None):
        self.primitive, self.invars, self.outvars, self.params = P(name), [], [], params or {}


class Jaxpr:
    def __init__(self, eqns): self.eqns, self.invars, self.outvars = eqns, [], []


class Closed:
    def __init__(self, jaxpr): self.jaxpr = jaxpr


class Canonical(unittest.TestCase):
    def test_layout_primitives_are_skipped(self):
        for n in ("reshape", "transpose", "convert_element_type", "broadcast_in_dim", "pjit", "squeeze"):
            self.assertIsNone(sg.canonical(n), n)

    def test_renames_and_suffix(self):
        self.assertEqual(sg.canonical("integer_pow"), "pow")
        self.assertEqual(sg.canonical("logistic"), "sigmoid")
        self.assertEqual(sg.canonical("dot_general_p"), "dot_general")
        self.assertEqual(sg.canonical("Reduce_Sum"), "reduce_sum")

    def test_walk_descends_into_sub_jaxprs_but_not_kernels(self):
        inner = Jaxpr([Eqn("dot_general"), Eqn("add")])
        body = Jaxpr([Eqn("mul"), Eqn("mul")])
        j = Jaxpr([Eqn("pjit", {"jaxpr": Closed(inner)}), Eqn("reshape"), Eqn("pallas_call", {"jaxpr": body}), Eqn("max")])
        self.assertEqual(sg.walk_ops(j), ["dot_general", "add", "pallas_call", "max"])

    def test_collapse_and_multiset(self):
        ops = ["add", "add", "mul", "add"]
        self.assertEqual(sg.collapse(ops), ["add", "mul", "add"])
        self.assertEqual(sg.multiset(ops), {"add": 3, "mul": 1})

    def test_jaccard(self):
        self.assertAlmostEqual(sg.jaccard({"a": 2, "b": 1}, {"a": 1, "c": 1}), 1 / 4)
        self.assertEqual(sg.jaccard({}, {}), 0.0)
        self.assertEqual(sg.jaccard({"a": 1}, {"a": 1}), 1.0)


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class Trace(unittest.TestCase):
    def test_41k_signature(self):
        code = (ROOT / "benchmarks/jaxbench/level1/41k_Gemm_Add_ReLU/baseline.py").read_text()
        s = sg.signature_of(code, PY)
        self.assertNotIn("error", s, s)
        self.assertIn("dot_general", s["op_chain"])
        self.assertTrue({"max", "relu"} & set(s["op_chain"]), s["op_chain"])
        self.assertEqual(s["input_shapes"][0], [4096, 8192])
        self.assertEqual(s["output_shapes"], [[4096, 8192]])

    def test_broken_module_reports_error(self):
        s = sg.signature_of("def workload(x): return x\n", PY)  # no create_inputs
        self.assertIn("error", s)

    def test_exam_cache_covers_most_tasks(self):
        sigs = sg.exam_signatures(python=PY)  # uses the cache when present
        self.assertEqual(len(sigs), 50)
        ok = sum("error" not in s for s in sigs.values())
        self.assertGreaterEqual(ok, 40, {k: v["error"][:80] for k, v in sigs.items() if "error" in v})


if __name__ == "__main__":
    unittest.main()
