"""The repair hold rule: index-output and quantized problems that are wrong on every seed get no repair and are listed for a
reference check; everything else repairs as usual."""
import pathlib, sys, tempfile, unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import generate  # noqa: E402
from common import read_jsonl, write_jsonl  # noqa: E402


def prob(task, chain=("dot_general",), dtypes=("bfloat16",)):
    return {"problem_id": "p/" + task, "task_name": task, "signature": {"op_chain": list(chain), "input_dtypes": list(dtypes)}}


def row(verdict="wrong", oks=(False, False, False)):
    return {"verdict": verdict, "per_seed": [{"seed": i, "ok": ok, "err_kernel": 0.5, "err_xla_default": 1e-3} for i, ok in enumerate(oks)]}


class HoldReason(unittest.TestCase):
    def test_index_output_by_op_chain(self):
        self.assertEqual(generate.hold_reason(prob("authored__moe__router_x", chain=("dot_general", "top_k")), row()), "index_output")

    def test_index_output_by_name(self):
        self.assertEqual(generate.hold_reason(prob("authored__moe__capacity_dispatch_combine"), row()), "index_output")

    def test_quantized_by_dtype(self):
        self.assertEqual(generate.hold_reason(prob("authored__q__w8a8_matmul", dtypes=("bfloat16", "int8")), row()), "quantized")

    def test_wrong_on_some_seeds_repairs(self):
        self.assertIsNone(generate.hold_reason(prob("authored__moe__router_x", chain=("top_k",)), row(oks=(True, False, False))))

    def test_other_verdicts_and_plain_problems_repair(self):
        self.assertIsNone(generate.hold_reason(prob("kb__l1__gemm"), row()))
        self.assertIsNone(generate.hold_reason(prob("authored__moe__router_x", chain=("top_k",)), row(verdict="compile")))
        self.assertIsNone(generate.hold_reason(prob("authored__moe__router_x", chain=("top_k",)), None))

    def test_int32_metadata_is_not_quantized(self):
        self.assertIsNone(generate.hold_reason(prob("authored__paged__kv_write", dtypes=("bfloat16", "int32")), row()))


class PlanRepair(unittest.TestCase):
    def test_holds_index_problem_and_repairs_the_rest_once(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            probs = {p["problem_id"]: p for p in (prob("authored__moe__router_x", chain=("top_k",)), prob("kb__l1__gemm"))}
            cands = {f"{p['task_name']}.a1.s0": {"candidate_id": f"{p['task_name']}.a1.s0", "task_name": p["task_name"], "attempt": 1,
                                                 "code": "def workload(x):\n    return x\n", "constructor": {"backend": "gemini"}} for p in probs.values()}
            write_jsonl(d / "traj" / "g1.jsonl", [{"problem_id": pid, "status": "open", "attempts": [
                {"candidate_id": f"{p['task_name']}.a1.s0", "attempt": 1, "sample": 0, "job_id": "dataset-g1-r1", "oracle_sha": "s",
                 "verdict": "wrong", "feedback": "wrong output on 3 of 3 seeds"}]} for pid, p in probs.items()])
            write_jsonl(d / "verdicts" / "dataset-g1-r1.jsonl", [{"candidate_id": c, "row": row()} for c in cands])
            with mock.patch.object(generate, "TRAJECTORIES", d / "traj"), mock.patch.object(generate, "VERDICTS", d / "verdicts"), \
                 mock.patch.object(generate, "HELD", d / "held.jsonl"):
                jobs = generate.plan_repair("g1", 2, probs, cands, None)
                again = generate.plan_repair("g1", 2, probs, cands, None)
            self.assertEqual([j[0]["task_name"] for j in jobs], ["kb__l1__gemm"])
            self.assertEqual([j[0]["task_name"] for j in again], ["kb__l1__gemm"])
            held = read_jsonl(d / "held.jsonl")
            self.assertEqual([(h["task_name"], h["reason"]) for h in held], [("authored__moe__router_x", "index_output")])
            self.assertEqual(len(held[0]["per_seed"]), 3)


if __name__ == "__main__":
    unittest.main()
