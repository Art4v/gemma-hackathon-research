"""assemble.py and verify_dataset.py on the synthetic fixtures after all three rounds are collected."""
import importlib, json, pathlib, shutil, sys, unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
from test_collect import fresh_root, run_collect  # noqa: E402
assemble = importlib.import_module("assemble")
verify = importlib.import_module("verify_dataset")
from common import read_json, read_jsonl, write_jsonl  # noqa: E402


class AssembleTests(unittest.TestCase):
    def setUp(self):
        self.root, self.oracle = fresh_root()
        for job in ("dataset-pilot-r1", "dataset-pilot-r2", "dataset-pilot-r3"):
            self.assertEqual(run_collect(self.root, self.oracle, job), 0, job)
        self.final = self.root / "final" / "pilot"

    def tearDown(self):
        shutil.rmtree(self.root.parent, ignore_errors=True)

    def run_assemble(self, *flags):
        return assemble.main(["--phase", "pilot", "--root", str(self.root), *flags])

    def examples(self):
        return {e["example_id"]: e for e in read_jsonl(self.final / "train.jsonl") + read_jsonl(self.final / "val.jsonl")}

    def test_emission_rules(self):
        self.assertEqual(self.run_assemble(), 0)
        ex = self.examples()
        kinds = {k: sum(1 for e in ex.values() if e["kind"] == k) for k in ("single", "repair")}
        # singles: p1 a1, p8 a1, p5 a1, p6 a1, p6 human, and the derived singles of p2, p3, p7; repairs: p2, p3, p7
        self.assertEqual(kinds, {"single": 8, "repair": 3}, sorted(ex))
        # a1 pass: single only, no derived_from
        self.assertIsNone(ex["kb__l1__01_Relu_Scale.single.a1"]["derived_from"])
        # a2 pass: repair + derived single
        r2 = ex["kb__l1__02_Gelu_Add.repair.a2"]
        self.assertEqual(ex["kb__l1__02_Gelu_Add.single.a2"]["derived_from"], r2["example_id"])
        self.assertEqual([t["type"] for t in r2["turns"]], ["problem", "kernel", "oracle_feedback", "kernel"])
        self.assertEqual(r2["n_repairs"], 1)
        self.assertEqual(r2["meta"]["failing_verdicts"], ["wrong"])
        # a3 pass: 6 turns, feedback verbatim from the verdict records
        r3 = ex["kb__l1__03_Norm_Mul.repair.a3"]
        self.assertEqual(len(r3["turns"]), 6)
        verdicts = {v["candidate_id"]: v for p in (self.root / "verdicts").glob("*.jsonl") for v in read_jsonl(p)}
        for i, t in enumerate(r3["turns"]):
            if t["type"] == "oracle_feedback":
                cid = r3["turns"][i - 1]["candidate_id"]
                self.assertEqual(t["feedback"], verdicts[cid]["feedback"])
                self.assertEqual(t["verdict_ref"]["job_id"], verdicts[cid]["job_id"])
        self.assertEqual(r3["meta"]["failing_verdicts"], ["compile", "vmem"])
        self.assertNotIn("kb__l1__03_Norm_Mul.repair.a3.suffix", ex)
        # correct_slow appears only as a failing turn
        r7 = ex["kb__l2__07_Slow_First.repair.a2"]
        self.assertEqual(r7["turns"][2]["verdict"], "correct_slow")
        self.assertTrue(all(e["final_verdict"] == "pass" for e in ex.values()))
        # harness_oom consumed no attempt: p5 is a plain single at a1
        self.assertIn("authored__attention__05_swa.single.a1", ex)
        self.assertEqual(ex["authored__attention__05_swa.single.a1"]["n_repairs"], 0)
        # human kernel
        h = ex["scraped__jax__06_example.single.human"]
        self.assertEqual(h["meta"]["target_origin"], "human")
        self.assertEqual(ex["scraped__jax__06_example.single.a1"]["meta"]["target_origin"], "constructor")
        # parity flag
        self.assertTrue(ex["kb__l1__01_Relu_Scale.single.a1"]["at_parity"])
        self.assertFalse(ex["kb__l1__03_Norm_Mul.single.a3"]["at_parity"])
        # unsolved and reverify
        unsolved = read_jsonl(self.final / "unsolved.jsonl")
        self.assertEqual([u["problem_id"] for u in unsolved], ["kb/l1/04_Hard"])
        reverify = read_jsonl(self.final / "reverify_items.jsonl")
        self.assertEqual(len(reverify), 8)  # 7 constructor final kernels + 1 human
        stats = read_json(self.final / "stats.json")
        self.assertEqual(stats["by_kind"], {"single": 8, "repair": 3})
        self.assertEqual(stats["pass_at_k"]["all"], {"pass@1": 0.5, "pass@2": 0.75, "pass@3": 0.875})
        self.assertEqual(stats["n_unsolved"], 1)

    def test_flags(self):
        self.assertEqual(self.run_assemble("--emit-suffix", "--no-human-targets"), 0)
        ex = self.examples()
        self.assertIn("kb__l1__03_Norm_Mul.repair.a3.suffix", ex)
        self.assertEqual(len(ex["kb__l1__03_Norm_Mul.repair.a3.suffix"]["turns"]), 4)
        self.assertNotIn("scraped__jax__06_example.single.human", ex)

    def test_split_is_deterministic_by_family(self):
        self.assertEqual(self.run_assemble("--val-frac", "0.5"), 0)
        ex = self.examples()
        a, b = ex["kb__l1__01_Relu_Scale.single.a1"], ex["kb__l1__01_Relu_Scale__v2.single.a1"]
        self.assertEqual(a["family_id"], b["family_id"])
        self.assertEqual(a["split"], b["split"])
        first = {k: e["split"] for k, e in ex.items()}
        self.assertEqual(self.run_assemble("--val-frac", "0.5"), 0)
        self.assertEqual({k: e["split"] for k, e in self.examples().items()}, first)

    def test_verify_passes_then_catches_corruption(self):
        self.assertEqual(self.run_assemble("--val-frac", "0.5"), 0)
        self.assertEqual(verify.main([str(self.final), "--root", str(self.root)]), 0)
        for name in ("train.jsonl", "val.jsonl"):
            rows = read_jsonl(self.final / name)
            for e in rows:
                if e["kind"] == "repair":
                    e["turns"][2]["feedback"] = e["turns"][2]["feedback"] + " (edited)"
                    write_jsonl(self.final / name, rows)
                    self.assertEqual(verify.main([str(self.final), "--root", str(self.root)]), 1)
                    return
        self.fail("no repair example found to corrupt")

    def test_verify_catches_exam_leak_and_missing_verdict(self):
        self.assertEqual(self.run_assemble(), 0)
        p = self.root / "problems" / "problems.jsonl"
        probs = read_jsonl(p)
        probs[0]["source_ref"]["kb_stem"] = "76_Gemm_Add_ReLU"  # the stem of exam task 41k
        write_jsonl(p, probs)
        self.assertEqual(verify.main([str(self.final), "--root", str(self.root)]), 1)
        write_jsonl(p, read_jsonl(p)[:0] + [dict(x, source_ref={k: v for k, v in x["source_ref"].items() if k != "kb_stem"}) if i == 0 else x
                                           for i, x in enumerate(probs)])
        self.assertEqual(verify.main([str(self.final), "--root", str(self.root)]), 0)
        (self.root / "verdicts" / "dataset-pilot-r3.jsonl").unlink()
        self.assertEqual(verify.main([str(self.final), "--root", str(self.root)]), 1)


if __name__ == "__main__":
    unittest.main()
