"""The CPU reject-only lowering pre-filter's bookkeeping (clanker, issue #1 6713940244). No JAX, no sandbox: the runner is a stand-in.
The real validation (sandboxed, JAX 0.10.2) is `jaxbench/lower_filter.py validate`; its result is posted on issue #1."""
import json, pathlib, sys, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import lower_filter as lf  # noqa: E402


class Outcome(unittest.TestCase):
    def test_only_a_candidate_failure_is_a_rejection(self):
        self.assertEqual(lf.outcome({"lowered": True}), "lowered")
        for stage in ("cand", "lower"):
            self.assertEqual(lf.outcome({"lowered": False, "stage": stage, "error": "ValueError: x"}), "rejected")
        for stage in ("setup", "ref", "child", None):  # a harness problem is never the kernel's rejection
            self.assertEqual(lf.outcome({"lowered": False, "stage": stage, "error": "x"}), "abstain")

    def test_the_child_result_shape_is_enforced(self):
        self.assertEqual(lf.child_result('{"lowered": true}', 0), {"lowered": True})
        ok = {"lowered": False, "stage": "lower", "error": "ValueError: block"}
        self.assertEqual(lf.child_result(json.dumps(ok), 0), ok)
        for bad in ("", "not json", '{"lowered": false, "stage": "elsewhere", "error": "x"}', '{"lowered": "yes"}', "[1]"):
            self.assertEqual(lf.outcome(lf.child_result(bad, 1)), "abstain", bad)


class Items(unittest.TestCase):
    TRAIN = [{"key": "gen_a/1", "messages": [{"content": "sys"}, {"content": "Reference:\n```python\ndef create_inputs(): ...\n```\nGo."},
                                             {"content": "```python\ndef workload(x): ...\n```"}]}]

    def test_label_items_carry_code_the_rows_reference_and_the_labels_full_size_shapes(self):
        labels = [{"meta": True}, {"id": "r2_ready/gen_a/1", "ran": True, "shapes": [[8192, 1024], [8192, 1024]]},
                  {"id": "r2_ready/gen_a/1", "ran": None, "shapes": [[1]]}, {"id": "r2_ready/missing/9", "ran": False, "shapes": [[1]]}]
        got = lf.label_items(labels, self.TRAIN)
        self.assertEqual(len(got), 1)
        i, payload, chip = got[0]
        self.assertEqual((i, chip, payload["shapes"]), ("r2_ready/gen_a/1", "ran", [[8192, 1024], [8192, 1024]]))
        self.assertEqual(payload["cand"].strip(), "def workload(x): ...")
        self.assertEqual(payload["ref"].strip(), "def create_inputs(): ...")  # dtypes come from here: an index input is an integer

    def test_study_items_take_ran_and_failed_verdicts_only_once_each(self):
        q = [{"id": f"k{n}", "code": "c", "ref_code": "r"} for n in range(6)]
        res = [{"id": "k0", "verdict": "pass"}, {"id": "k1", "verdict": "correct_slow"}, {"id": "k2", "verdict": "compile"},
               {"id": "k3", "verdict": "not_a_kernel"}, {"id": "k4", "verdict": "harness_oom"}, {"id": "k0", "verdict": "pass"},
               {"id": "nope", "verdict": "pass"}]
        got = {i: chip for i, _, chip in lf.study_items(res, q)}
        self.assertEqual(got, {"k0": "ran", "k1": "ran", "k2": "failed"})  # not_a_kernel, harness_oom say nothing about lowering


class Bar(unittest.TestCase):
    def test_one_wrong_rejection_fails_the_bar(self):
        rows = [("a", "ran", {"lowered": True}), ("b", "failed", {"lowered": False, "stage": "lower", "error": "ValueError: block"}),
                ("c", "failed", {"lowered": True})]
        s = lf.summarise(rows)
        self.assertEqual((s["bar_met"], s["wrongly_rejected"], s["chip_failures_caught"], s["chip_failures"]), (True, 0, 1, 2))
        s = lf.summarise(rows + [("d", "ran", {"lowered": False, "stage": "lower", "error": "VerificationError: x"})])
        self.assertEqual((s["bar_met"], s["wrongly_rejected_ids"][0]["id"]), (False, "d"))

    def test_an_abstention_on_a_ran_kernel_is_not_a_wrong_rejection(self):
        s = lf.summarise([("a", "ran", {"lowered": False, "stage": "ref", "error": "x"})])
        self.assertTrue(s["bar_met"])

    def test_validate_runs_every_item_through_the_runner(self):
        items = [("a", {"cand": "ok"}, "ran"), ("b", {"cand": "bad"}, "failed")]
        run = lambda p: {"lowered": True} if p["cand"] == "ok" else {"lowered": False, "stage": "lower", "error": "E"}  # noqa: E731
        rows, s = lf.validate(items, run, workers=2)
        self.assertEqual(([r[0] for r in rows], s["table"]), (["a", "b"], {"failed/rejected": 1, "ran/lowered": 1}))


class Filter(unittest.TestCase):
    # clanker, issue #1 6715359660: run it on a generate run's kernels before packing; the chip still judges every pass
    def test_each_kernel_with_code_gets_one_outcome_and_only_the_error_type_is_kept(self):
        items = [{"task": "gen_a", "ref": "REF_A"}, {"task": "gen_b", "ref": "REF_B"}]
        rows = [{"id": "gen_a/r0/t0/1", "task": "gen_a", "code": "CODE_A"}, {"id": "gen_b/r0/t0/2", "task": "gen_b", "code": "CODE_B"},
                {"id": "gen_b/r0/t0/3", "task": "gen_b", "code": ""}]
        seen = []
        def run(payload):
            seen.append(payload)
            return {"lowered": True} if payload["cand"] == "CODE_A" else {"lowered": False, "stage": "lower", "error": "ValueError: block shape (7, 100) secret code"}
        out = lf.filter_rows(items, rows, run, workers=1)
        self.assertEqual(sorted(seen, key=lambda p: p["cand"]), [{"cand": "CODE_A", "ref": "REF_A"}, {"cand": "CODE_B", "ref": "REF_B"}])
        self.assertEqual(out, [{"id": "gen_a/r0/t0/1", "task": "gen_a", "outcome": "lowered", "stage": None, "error_type": None},
                               {"id": "gen_b/r0/t0/2", "task": "gen_b", "outcome": "rejected", "stage": "lower", "error_type": "ValueError"}])

    def test_a_reference_failure_abstains(self):
        out = lf.filter_rows([{"task": "gen_a", "ref": "R"}], [{"id": "x", "task": "gen_a", "code": "C"}],
                             lambda p: {"lowered": False, "stage": "ref", "error": "ImportError: no"}, workers=1)
        self.assertEqual(out[0]["outcome"], "abstain")


class Feedback(unittest.TestCase):
    # clanker, issue #1 6715692985: rejected kernels go back to the teacher as an ordinary results row, committed, so the message is made safe
    BLOCK = ("ValueError: The Pallas TPU lowering currently requires that the last two dimensions of your block shape are divisible by 8 and "
             "128 respectively, or be equal to the respective dimensions of the overall array. In this case, the last two dimensions of "
             "the block shape are (4, 100) and the array shape is (1024, 1024).")

    def test_only_rejected_kernels_get_a_results_row_that_rounds_reads_as_a_verdict(self):
        rows = [{"id": "a", "task": "gen_a"}, {"id": "b", "task": "gen_b"}, {"id": "c", "task": "gen_c"}]
        res = [{"lowered": True}, {"lowered": False, "stage": "lower", "error": self.BLOCK}, {"lowered": False, "stage": "ref", "error": "X: y"}]
        out = lf.feedback_rows(rows, res)
        self.assertEqual([(r["id"], r["verdict"], r["judge"]) for r in out], [("b", "compile", "cpu_lowering")])  # lowered waits; abstain isn't feedback

    def test_the_numbers_survive_the_cap_and_paths_and_keys_do_not(self):
        fb = lf.feedback_text(self.BLOCK)
        self.assertIn("divisible by 8 and 128", fb)
        self.assertIn("(4, 100)", fb)  # the tail is kept: that's what the teacher needs to fix it
        long = "TypeError: " + "x " * 600 + "end of message"
        self.assertLessEqual(len(lf.feedback_text(long)), lf.FEEDBACK_CHARS + 80)
        self.assertIn("end of message", lf.feedback_text(long))
        leaky = ("AttributeError: module 'jax.experimental.pallas' has no attribute 'Unblocked' "
                 "(/Users/aarav/Documents/GitHub/planningGemma-jaxvenv/lib/python3.14/site-packages/jax/_src/x.py:12, "
                 "/private/tmp/claude-501/abc/lower_filter.py) token=ghp_abcdefghijklmnop1234567890")
        fb = lf.feedback_text(leaky)
        self.assertIn("has no attribute 'Unblocked'", fb)
        for gone in ("aarav", "/Users/", "/private/tmp", "ghp_abcdefghijklmnop1234567890", "claude-501"):
            self.assertNotIn(gone, fb, gone)

    def test_through_the_real_path_the_shapes_survive(self):
        # clanker, 6716525540: feedback_text was tested alone, while lower_sandboxed's scrub had already cut the message to ~200
        long_block = self.BLOCK.replace("In this case,", "In this case, " + "with some intervening detail, " * 20)
        self.assertGreater(len(long_block), 700)
        stdout = json.dumps({"lowered": False, "stage": "lower", "error": lf._bound(long_block)})
        r = lf._post(lf.child_result(stdout, 0))
        self.assertLessEqual(len(r["error"]), 210)  # the report's copy stays short
        self.assertIn("(4, 100)", r["message"])
        fb = lf.feedback_rows([{"id": "a", "task": "gen_a"}], [r])[0]["feedback"]
        self.assertIn("(4, 100)", fb)
        self.assertIn("(1024, 1024)", fb)
        self.assertIn("divisible by 8 and 128", fb)

    def test_the_child_keeps_head_and_tail_not_only_the_head(self):
        msg = "ValueError: head " + "x" * 5000 + " tail (4, 100)"
        b = lf._bound(msg)
        self.assertTrue(b.startswith("ValueError: head") and b.endswith("tail (4, 100)"))
        self.assertLessEqual(len(b), 2 * lf.CHILD_KEEP + 5)
        self.assertEqual(lf._bound("short"), "short")

    def test_a_token_in_the_message_is_redacted_before_it_is_kept(self):
        r = lf._post({"lowered": False, "stage": "lower", "error": "TypeError: bad ghp_abcdefghijklmnop1234567890 shape (8, 128)"})
        self.assertNotIn("ghp_abcdefghijklmnop1234567890", r["message"])
        self.assertIn("(8, 128)", r["message"])

    def test_the_rounds_side_reads_it_unchanged(self):
        sys.path.insert(0, str(lf.HERE))
        try:
            from jaxbench import rounds  # the teacher loop, which this repository doesn't have yet
        except ImportError:
            self.skipTest("the teacher loop (rounds.py) isn't in this repository yet")
        row = {"id": rounds.row_id("gen_a", 0, 0, "K"), "task": "gen_a", "rollout": 0, "round": 0, "code": "K", "history": []}
        fb = lf.feedback_rows([row], [{"lowered": False, "stage": "lower", "error": self.BLOCK}])
        jobs = rounds.pending([{"task": "gen_a", "ref": "R"}], [row], rounds.read_verdicts(fb, [row]), 1, 1)
        self.assertEqual((len(jobs), jobs[0]["round"]), (1, 1))
        self.assertIn("divisible by 8 and 128", jobs[0]["history"][0]["feedback"])


if __name__ == "__main__":
    unittest.main()
