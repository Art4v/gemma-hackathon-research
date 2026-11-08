"""Unit tests for the pieces of the pipeline that need no constructor and no JAX: the renderer round trip, the parse check, the
submission selector, the candidate-id scheme and the record validators."""
import pathlib, sys, unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import common  # noqa: E402
import oracle_submit  # noqa: E402
import prescreen  # noqa: E402
import render_chat  # noqa: E402

REF = "import jax, jax.numpy as jnp\nCONFIG = {'name': 'toy', 'm': 16, 'n': 128}\ndef create_inputs():\n    return (jnp.ones((16, 128)),)\ndef workload(x):\n    return x * 2.0\n"
KER = "import jax, jax.numpy as jnp\nfrom jax.experimental import pallas as pl\ndef _k(x_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2.0\ndef workload(x):\n    return pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)\n"


def cand(task, attempt, sample, backend="claude_cli", code=KER):
    return {"candidate_id": common.candidate_id(task, attempt, sample), "problem_id": "p/" + task, "task_name": task, "attempt": attempt, "sample": sample,
            "parent_candidate_id": None if attempt == 1 else common.candidate_id(task, attempt - 1, 0), "feedback_in": None if attempt == 1 else "wrong output",
            "feedback_in_verdict": None if attempt == 1 else {"job_id": "j", "oracle_sha": "s", "verdict": "wrong"}, "prescreen_hint": None,
            "constructor": {"backend": backend, "model": "m", "prompt_version": "v", "docs_sha256": None}, "plan": "p", "code": code,
            "code_sha256": common.sha256(code), "usage": {}, "created_at": common.now()}


class Render(unittest.TestCase):
    def test_round_trip_single_and_repair(self):
        ex = {"turns": [{"type": "problem", "ref_code": REF}, {"type": "kernel", "code": KER}, {"type": "oracle_feedback", "feedback": "wrong output on 1 of 3 seeds"},
                        {"type": "kernel", "code": KER + "# v2\n"}]}
        msgs = render_chat.render(ex)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])
        self.assertEqual(render_chat.unfence(msgs[1]["content"]), KER)
        self.assertEqual(render_chat.unfence(msgs[3]["content"]), KER + "# v2\n")
        self.assertTrue(msgs[2]["content"].startswith("Oracle verdict: wrong output on 1 of 3 seeds"))
        self.assertIn(REF.rstrip("\n"), msgs[0]["content"])

    def test_unfence_handles_bare_module_and_rejects_two_fences(self):
        self.assertEqual(render_chat.unfence(KER), KER)
        self.assertIsNone(render_chat.unfence("```python\na=1\n```\n```python\nb=2\n```"))


class Parse(unittest.TestCase):
    def test_parse_check(self):
        self.assertIsNone(prescreen.parse_check(KER))
        self.assertIn("syntax", prescreen.parse_check("def workload(:\n"))
        self.assertIn("workload", prescreen.parse_check("import jax\nfrom jax.experimental import pallas\n"))
        self.assertIn("pallas", prescreen.parse_check("def workload(x):\n    return x\n"))


class Select(unittest.TestCase):
    def test_latest_passing_sample_wins_and_probes_are_listed(self):
        cs = [cand("t", 1, 0), cand("t", 1, 1), cand("t", 1, 2), cand("h", 1, 0, backend="human")]
        cs[3]["candidate_id"] = "h.human.s0"
        pres = {"t.a1.s0": {"outcome": "reject"}, "t.a1.s1": {"outcome": "pass"}, "t.a1.s2": {"outcome": "reject"}}
        chosen, probes = oracle_submit.select(cs, pres, probes=False)
        self.assertEqual(sorted(c["candidate_id"] for c in chosen), ["h.human.s0", "t.a1.s1"])
        self.assertEqual(probes, [])
        chosen, probes = oracle_submit.select(cs, pres, probes=True)
        self.assertEqual(sorted(probes), ["t.a1.s0", "t.a1.s2"])
        self.assertEqual(len(chosen), 4)

    def test_all_rejected_submits_the_latest_anyway(self):
        cs = [cand("t", 1, 0), cand("t", 1, 1)]
        pres = {"t.a1.s0": {"outcome": "reject"}, "t.a1.s1": {"outcome": "reject"}}
        chosen, _ = oracle_submit.select(cs, pres, probes=False)
        self.assertEqual([c["candidate_id"] for c in chosen], ["t.a1.s1"])


class Records(unittest.TestCase):
    def test_candidate_validation(self):
        self.assertEqual(common.validate("candidate", cand("t", 1, 0)), [])
        self.assertEqual(common.validate("candidate", cand("t", 2, 0)), [])
        bad = cand("t", 2, 0); bad["parent_candidate_id"] = None
        self.assertTrue(any("parent" in e for e in common.validate("candidate", bad)))

    def test_task_name_and_exam_stems(self):
        self.assertEqual(common.task_name("kb/l2/63_Gemm_ReLU_Divide"), "kb__l2__63_Gemm_ReLU_Divide")
        stems = common.exam_kb_stems()
        self.assertEqual(len(stems), 33)
        self.assertEqual(stems["76_Gemm_Add_ReLU"], "41k_Gemm_Add_ReLU")


if __name__ == "__main__":
    unittest.main()
