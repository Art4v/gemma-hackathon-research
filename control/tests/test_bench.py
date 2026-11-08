"""control/bench.py: the attempt loop, with a fake model and fake oracle rows. No JAX, no Ollama."""
import json, pathlib, sys, tempfile, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bench  # noqa: E402

T = {"1p_A": ("def workload(x):\n    return x\n", "jaxbench/level1/1p_A"), "2k_B": ("def workload(x):\n    return x * 2\n", "jaxbench/level1/2k_B")}


def fake_chat(messages, seed):
    n = sum(m["role"] == "assistant" for m in messages)
    return {"message": {"content": f"Sure.\n```python\n# attempt {n} seed {seed}\ndef workload(x):\n    return x\n```\n"},
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20}


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = bench.RUNS, bench.tasks
        bench.RUNS, bench.tasks = pathlib.Path(self.tmp.name), lambda: dict(T)

    def tearDown(self):
        bench.RUNS, bench.tasks = self.old
        self.tmp.cleanup()

    def verdict(self, id_, verdict, feedback="", speed=None):
        p = bench.run_dir(bench.MODEL) / "verdicts.jsonl"
        with p.open("a") as f:
            f.write(json.dumps({"id": id_, "verdict": verdict, "feedback": feedback, "speed_vs_xla": speed}) + "\n")

    def test_turn_0_for_every_task_then_nothing_until_the_oracle_answers(self):
        self.assertEqual(bench.generate("bare", bench.MODEL, chat_fn=fake_chat, log=lambda *_: None), 2)
        self.assertEqual(bench.generate("bare", bench.MODEL, chat_fn=fake_chat, log=lambda *_: None), 0)  # waiting for verdicts
        self.assertEqual([i["id"] for i in bench.items(bench.MODEL)], ["bare/1p_A/t0", "bare/2k_B/t0"])

    def test_a_retry_sees_every_earlier_attempt_with_the_oracles_feedback(self):
        bench.generate("docs", bench.MODEL, ["1p_A"], chat_fn=fake_chat, log=lambda *_: None)
        self.verdict("docs/1p_A/t0", "compile", "compile error: no workload")
        seen = []
        bench.generate("docs", bench.MODEL, ["1p_A"], chat_fn=lambda m, s: seen.append(m) or fake_chat(m, s), log=lambda *_: None)
        m = seen[0]
        self.assertEqual([x["role"] for x in m], ["system", "user", "assistant", "user"])
        self.assertTrue(m[1]["content"].startswith("Pallas documentation:\n"))  # the docs arm
        self.assertEqual(m[3]["content"], "compile error: no workload\nReply with the corrected Python code block.")
        self.assertIn("# attempt 0", m[2]["content"])

    def test_a_task_stops_at_pass_terminal_verdicts_and_after_three_turns(self):
        bench.generate("bare", bench.MODEL, chat_fn=fake_chat, log=lambda *_: None)
        self.verdict("bare/1p_A/t0", "pass", "pass: 1.2x XLA", 1.2)
        self.verdict("bare/2k_B/t0", "wrong", "wrong output")
        self.assertEqual(bench.generate("bare", bench.MODEL, chat_fn=fake_chat, log=lambda *_: None), 1)  # only 2k_B retries
        self.verdict("bare/2k_B/t1", "harness_oom", "harness error")
        self.assertEqual(bench.generate("bare", bench.MODEL, chat_fn=fake_chat, log=lambda *_: None), 0)
        rows = bench.read_jsonl(bench.run_dir(bench.MODEL) / "bare.jsonl")
        self.assertEqual(sorted(r["id"] for r in rows), ["bare/1p_A/t0", "bare/2k_B/t0", "bare/2k_B/t1"])

    def test_at_most_three_turns(self):
        for turn in range(bench.TURNS):
            bench.generate("bare", bench.MODEL, ["2k_B"], chat_fn=fake_chat, log=lambda *_: None)
            self.verdict(f"bare/2k_B/t{turn}", "wrong", f"wrong {turn}")
        self.assertEqual(bench.generate("bare", bench.MODEL, ["2k_B"], chat_fn=fake_chat, log=lambda *_: None), 0)

    def test_both_arms_get_the_same_seeds_and_rows_record_the_settings(self):
        for arm in bench.ARMS:
            bench.generate(arm, bench.MODEL, ["1p_A"], chat_fn=fake_chat, log=lambda *_: None)
        b, d = (bench.read_jsonl(bench.run_dir(bench.MODEL) / f"{arm}.jsonl")[0] for arm in bench.ARMS)
        self.assertEqual(b["seed"], d["seed"])
        self.assertNotEqual(b["prompt_sha256"], d["prompt_sha256"])
        for k, v in {"temperature": 0, "thinking": False, "num_ctx": bench.NUM_CTX, "max_tokens": bench.MAX_TOKENS, "finish": "stop"}.items():
            self.assertEqual(b[k], v, k)
        self.assertIn("def workload", b["code"])

    def test_ingest_refuses_rows_that_are_not_this_runs_attempts(self):
        bench.generate("bare", bench.MODEL, ["1p_A"], chat_fn=fake_chat, log=lambda *_: None)
        p = pathlib.Path(self.tmp.name) / "r.jsonl"
        p.write_text(json.dumps({"id": "bare/9p_X/t0", "verdict": "pass"}) + "\n")
        with self.assertRaises(SystemExit):
            bench.ingest(bench.MODEL, [p])

    def test_summary_counts_pass_at_1_and_within_3(self):
        bench.generate("bare", bench.MODEL, chat_fn=fake_chat, log=lambda *_: None)
        self.verdict("bare/1p_A/t0", "pass", "", 1.5)
        self.verdict("bare/2k_B/t0", "compile", "c")
        bench.generate("bare", bench.MODEL, chat_fn=fake_chat, log=lambda *_: None)
        self.verdict("bare/2k_B/t1", "correct_slow", "slow", 0.5)
        s = bench.summary(bench.MODEL)["bare"]
        self.assertEqual((s["pass_at_1"], s["pass_within_3"], s["correct_within_3"], s["parity_within_3"]), (1, 1, 2, 1))
        self.assertEqual(s["median_speed_correct"], 1.0)


class Prompt(unittest.TestCase):
    def test_extract_code_takes_the_last_closed_block_and_nothing_from_an_unclosed_one(self):
        self.assertEqual(bench.extract_code("a\n```python\nx = 1\n```\nb\n```python\ny = 2\n```"), "y = 2\n")
        self.assertEqual(bench.extract_code("```python\nx = 1\n# looping forever"), "")

    def test_the_wording_is_planninggemmas(self):
        m = bench.conversation("REF", [], None)
        self.assertEqual(m[1]["content"], "Here is the reference JAX function:\n```python\nREF\n```\nPlease convert this function into a Pallas TPU kernel.")
        self.assertTrue(m[0]["content"].startswith("You are a code model that writes TPU kernels in JAX Pallas."))


if __name__ == "__main__":
    unittest.main()
