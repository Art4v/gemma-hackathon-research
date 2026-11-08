"""dataset/scripts/filter_contamination.py (layers 1-2, no tracing: signatures injected) and filter_neardup.py (layer 3)."""
import pathlib, re, sys, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dataset" / "scripts"))

import common  # noqa: E402
import filter_contamination as fc  # noqa: E402
import filter_neardup as fn  # noqa: E402

EXAM_41K = (ROOT / "benchmarks/jaxbench/level1/41k_Gemm_Add_ReLU/baseline.py").read_text()
STEMS = {"76_Gemm_Add_ReLU": "41k_Gemm_Add_ReLU", "88_Gemm_GroupNorm_Swish_Multiply_Swish": "45k_Gemm_GroupNorm_Swish_Multiply_Swish"}
SIG_41K = {"op_chain": ["dot_general", "add", "max"], "op_multiset": {"add": 1, "dot_general": 1, "max": 1},
           "input_shapes": [[4096, 8192], [8192, 8192], [8192]], "input_dtypes": ["float32"] * 3, "output_shapes": [[4096, 8192]],
           "input_bytes": 402685952}
EXAM_SIGS = {"41k_Gemm_Add_ReLU": SIG_41K, "12p_RMSNorm": {"op_chain": ["pow", "reduce_sum", "div", "add", "rsqrt", "mul"],
             "op_multiset": {"pow": 1, "reduce_sum": 1, "div": 1, "add": 1, "rsqrt": 1, "mul": 2}}, "99p_broken": {"error": "x"}}


def problem(pid, source="kernelbench", kb_stem=None, sig=None, ref_code="import jax\n", path=None):
    return {"problem_id": pid, "family_id": pid, "task_name": common.task_name(pid), "source": source,
            "source_ref": {k: v for k, v in (("kb_stem", kb_stem), ("path", path)) if v is not None}, "ref_code": ref_code,
            "ref_sha256": common.sha256(ref_code), "signature": sig, "input_bytes": None, "fits_hbm": None, "status": "ok",
            "exclude_reason": None, "pipeline_version": "t", "created_at": common.now()}


class NameLayer(unittest.TestCase):
    def test_exact_stem_is_excluded(self):
        self.assertEqual(fc.name_match("76_Gemm_Add_ReLU", STEMS), "41k_Gemm_Add_ReLU")

    def test_renumbered_stem_is_excluded(self):
        self.assertEqual(fc.name_match("3_Gemm_Add_ReLU", STEMS), "41k_Gemm_Add_ReLU")
        self.assertEqual(fc.name_match("gemm_add_relu", STEMS), "41k_Gemm_Add_ReLU")

    def test_other_stem_is_kept(self):
        self.assertIsNone(fc.name_match("63_Gemm_ReLU_Divide", STEMS))
        self.assertIsNone(fc.name_match(None, STEMS))

    def test_real_exam_stems_has_33_k_tasks(self):
        stems = common.exam_kb_stems()
        self.assertEqual(len(stems), 33, stems)
        self.assertEqual(stems["76_Gemm_Add_ReLU"], "41k_Gemm_Add_ReLU")


class OpChainLayer(unittest.TestCase):
    def test_run_excludes_by_name_then_opchain_and_reviews_near(self):
        probs = [problem("kb/l2/76_Gemm_Add_ReLU", kb_stem="76_Gemm_Add_ReLU", sig=SIG_41K),
                 problem("kb/l2/63_Gemm_ReLU_Divide", kb_stem="63_Gemm_ReLU_Divide", sig=SIG_41K),   # same chain as 41k, other name
                 problem("auth/norm/rms_residual", source="authored",
                         sig={"op_chain": ["pow", "reduce_sum", "div", "add", "rsqrt", "mul", "add"],
                              "op_multiset": {"pow": 1, "reduce_sum": 1, "div": 1, "add": 2, "rsqrt": 1, "mul": 2}, "input_bytes": 10}),
                 problem("auth/misc/softmax", source="authored",
                         sig={"op_chain": ["reduce_max", "sub", "exp", "reduce_sum", "div"], "op_multiset": {"reduce_max": 1, "sub": 1, "exp": 1, "reduce_sum": 1, "div": 1}, "input_bytes": 10})]
        out, excluded, review, sig_updates = fc.run(probs, STEMS, EXAM_SIGS, log=lambda *_: None)
        reasons = {p["problem_id"]: p["exclude_reason"] for p in out}
        self.assertEqual(reasons["kb/l2/76_Gemm_Add_ReLU"], "contamination:name:41k_Gemm_Add_ReLU")
        self.assertEqual(reasons["kb/l2/63_Gemm_ReLU_Divide"], "contamination:opchain:41k_Gemm_Add_ReLU")
        self.assertIsNone(reasons["auth/norm/rms_residual"])
        self.assertEqual([r["problem_id"] for r in review], ["auth/norm/rms_residual"])
        self.assertGreaterEqual(review[0]["jaccard"], 0.8)
        self.assertIsNone(reasons["auth/misc/softmax"])
        self.assertEqual(len(excluded), 2)
        self.assertIn("auth/misc/softmax", sig_updates)
        for p in out:
            self.assertEqual(common.validate("problem", p), [], p["problem_id"])


def rename(code, mapping):
    for a, b in mapping.items():
        code = re.sub(rf"\b{a}\b", b, code)
    return code


class NearDup(unittest.TestCase):
    def setUp(self):
        self.corpus = fn.exam_corpus()
        self.assertEqual(len(self.corpus), 73)

    def test_boilerplate_is_stripped(self):
        s = fn.strip_boilerplate(EXAM_41K)
        self.assertNotIn("def benchmark", s)
        self.assertNotIn("__main__", s)
        self.assertIn("def workload", s)

    def test_normalise_drops_docstrings_comments_and_literals(self):
        toks = fn.normalise('"""doc"""\nimport jax  # c\nX = 4096\ndef f(a):\n    """d"""\n    return a * 2.5 + "s"\n')
        self.assertNotIn('"""doc"""', toks)
        self.assertIn("N", toks)
        self.assertIn("S", toks)
        self.assertNotIn("4096", toks)
        self.assertIn("x", toks)  # identifiers lower-cased
        self.assertEqual(toks.count("S"), 1)  # the docstring d was dropped, the literal "s" kept

    def test_renamed_copy_of_exam_file_is_excluded(self):
        copy = rename(EXAM_41K, {"x": "inp", "weight": "w_mat", "bias": "b_vec", "key": "k0", "rand_key": "rk", "ka": "k1", "kb": "k2"})
        copy = copy.replace("76_Gemm_Add_ReLU", "my_gemm")
        probs = [problem("kb/l2/copy", ref_code=copy, sig=SIG_41K)]
        with tempfile.TemporaryDirectory() as d:
            out, excluded, _, worst = fn.run(probs, self.corpus, pathlib.Path(d) / "problems.jsonl", log=lambda *_: None)
        self.assertEqual(len(excluded), 1)
        self.assertTrue(out[0]["exclude_reason"].startswith("contamination:neardup:41k_Gemm_Add_ReLU"), out[0]["exclude_reason"])

    def test_unrelated_module_is_kept_and_duplicates_collapse(self):
        other = ("import jax, jax.numpy as jnp\nCONFIG={'name':'softmax_rows','rows':8192,'cols':4096}\n"
                 "def create_inputs():\n    k=jax.random.key(3)\n    return (jax.random.normal(k,(CONFIG['rows'],CONFIG['cols']),jnp.bfloat16),)\n"
                 "def workload(x):\n    m=jnp.max(x,axis=-1,keepdims=True)\n    e=jnp.exp(x-m)\n    return e/jnp.sum(e,axis=-1,keepdims=True)\n")
        sig = {"op_chain": ["reduce_max", "sub", "exp", "reduce_sum", "div"], "input_shapes": [[8192, 4096]], "input_dtypes": ["bfloat16"]}
        probs = [problem("auth/a", source="authored", ref_code=other, sig=sig), problem("auth/b", source="authored", ref_code=other + "\n", sig=sig)]
        with tempfile.TemporaryDirectory() as d:
            out, excluded, sig_updates, worst = fn.run(probs, self.corpus, pathlib.Path(d) / "problems.jsonl", log=lambda *_: None)
        self.assertEqual(out[0]["status"], "ok")
        self.assertEqual(out[1]["exclude_reason"], "duplicate:auth/a")
        self.assertIn("auth/a", sig_updates)
        self.assertEqual(len(sig_updates["auth/a"]["nearest_exam_text"]["ref"]), 3)
        self.assertLess(worst[0], fn.JACCARD_BAR)

    def test_sibling_kernel_file_is_compared(self):
        exam_kernel = (ROOT / "benchmarks/jaxbench/level2/8p_GEMM/baseline.py").read_text()
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            (d / "scraped").mkdir()
            (d / "scraped" / "gemm.kernel.py").write_text(exam_kernel)
            ref = "import jax, jax.numpy as jnp\nCONFIG={'name':'g','m':4096}\ndef create_inputs():\n    return (jnp.ones((4096,4096)),)\ndef workload(a):\n    return a @ a\n"
            probs = [problem("scr/gemm", source="scraped", ref_code=ref, path="scraped/gemm.py", sig={"op_chain": ["dot_general"], "input_shapes": [[4096, 4096]], "input_dtypes": ["float32"]})]
            out, excluded, _, _ = fn.run(probs, self.corpus, d / "problems.jsonl", log=lambda *_: None)
        self.assertEqual(len(excluded), 1)
        self.assertTrue(out[0]["exclude_reason"].endswith(":kernel"), out[0]["exclude_reason"])
        self.assertIn("8p_GEMM", out[0]["exclude_reason"])


if __name__ == "__main__":
    unittest.main()
