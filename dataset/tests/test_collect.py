"""oracle_collect.py on the synthetic fixtures: checks, verdict records, trajectory merging, idempotence, failure modes."""
import importlib, json, pathlib, shutil, sys, tempfile, unittest

HERE = pathlib.Path(__file__).resolve().parent
FIX = HERE / "fixtures"
sys.path.insert(0, str(HERE.parent / "scripts"))
collect = importlib.import_module("oracle_collect")
from common import read_jsonl, write_jsonl  # noqa: E402


def fresh_root():
    """A temp copy of the fixtures: (dataset root, oracle dir)."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="dataset-test-"))
    if not (FIX / "dataset_root").exists():
        sys.path.insert(0, str(FIX))
        importlib.import_module("build_fixtures").build(FIX)
    shutil.copytree(FIX / "dataset_root", d / "dataset_root")
    shutil.copytree(FIX / "oracle", d / "oracle")
    return d / "dataset_root", d / "oracle"


def run_collect(root, oracle, job, *flags):
    return collect.main([job, "--root", str(root), "--jobs-dir", str(oracle / "jobs"), "--results-dir", str(oracle / "results"), *flags])


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.root, self.oracle = fresh_root()

    def tearDown(self):
        shutil.rmtree(self.root.parent, ignore_errors=True)

    def traj(self):
        return {t["problem_id"]: t for t in read_jsonl(self.root / "trajectories" / "pilot.jsonl")}

    def test_round1_statuses_and_records(self):
        self.assertEqual(run_collect(self.root, self.oracle, "dataset-pilot-r1"), 0)
        verdicts = read_jsonl(self.root / "verdicts" / "dataset-pilot-r1.jsonl")
        self.assertEqual(len(verdicts), 9)  # p1 p8 p2 p3 p4 p5 p6-human p6 p7
        self.assertTrue(all(v["accepted"] == (v["verdict"] == "pass") for v in verdicts))
        t = self.traj()
        self.assertEqual(t["kb/l1/01_Relu_Scale"]["status"], "solved")
        self.assertEqual(t["kb/l1/02_Gelu_Add"]["status"], "open")
        self.assertEqual(t["authored/attention/05_swa"]["status"], "harness_blocked")
        self.assertEqual(t["authored/attention/05_swa"]["attempts"], [])
        self.assertEqual(len(t["authored/attention/05_swa"]["harness_events"]), 1)
        self.assertEqual(t["scraped/jax/06_example"]["human"]["verdict"], "pass")
        self.assertEqual(len(t["scraped/jax/06_example"]["attempts"]), 1)
        self.assertEqual(t["kb/l2/07_Slow_First"]["status"], "open")

    def test_idempotent_and_final_statuses(self):
        for job in ("dataset-pilot-r1", "dataset-pilot-r1", "dataset-pilot-r2", "dataset-pilot-r3", "dataset-pilot-r2"):
            self.assertEqual(run_collect(self.root, self.oracle, job), 0, job)
        t = self.traj()
        self.assertEqual({p: x["status"] for p, x in t.items()},
                         {"kb/l1/01_Relu_Scale": "solved", "kb/l1/02_Gelu_Add": "solved", "kb/l1/03_Norm_Mul": "solved",
                          "kb/l1/04_Hard": "unsolved", "authored/attention/05_swa": "solved", "scraped/jax/06_example": "solved",
                          "kb/l2/07_Slow_First": "solved", "kb/l1/01_Relu_Scale__v2": "solved"})
        self.assertEqual([a["attempt"] for a in t["kb/l1/03_Norm_Mul"]["attempts"]], [1, 2, 3])
        self.assertEqual([a["verdict"] for a in t["kb/l1/03_Norm_Mul"]["attempts"]], ["compile", "vmem", "pass"])
        self.assertEqual(len(t["kb/l1/04_Hard"]["attempts"]), 3)
        p5 = t["authored/attention/05_swa"]
        self.assertEqual([(a["attempt"], a["sample"], a["verdict"]) for a in p5["attempts"]], [(1, 1, "pass")])
        self.assertEqual(len(p5["harness_events"]), 1)

    def test_expect_pass_fails_on_mixed_round(self):
        self.assertEqual(run_collect(self.root, self.oracle, "dataset-pilot-r1", "--expect-pass"), 1)

    def test_tampered_reference_sha_fails(self):
        p = self.oracle / "results" / "dataset-pilot-r2" / "results" / "dataset-pilot-r2.jsonl"
        rows = read_jsonl(p)
        rows[1]["ref_sha256"] = "0" * 64
        write_jsonl(p, rows)
        self.assertEqual(run_collect(self.root, self.oracle, "dataset-pilot-r2"), 1)
        self.assertFalse((self.root / "verdicts" / "dataset-pilot-r2.jsonl").exists())

    def test_missing_row_and_wrong_bar_fail(self):
        p = self.oracle / "results" / "dataset-pilot-r3" / "results" / "dataset-pilot-r3.jsonl"
        rows = read_jsonl(p)
        rows[1]["speed_bar"] = 0.8
        write_jsonl(p, rows[:-1])
        self.assertEqual(run_collect(self.root, self.oracle, "dataset-pilot-r3"), 1)

    def test_manifest_check_runs_for_packed_job(self):
        m = self.oracle / "jobs" / "dataset-pilot-r1" / "MANIFEST.json"
        d = json.loads(m.read_text())
        d["oracle_sha"] = "ffffffffffff"
        m.write_text(json.dumps(d))
        self.assertEqual(run_collect(self.root, self.oracle, "dataset-pilot-r1"), 1)


if __name__ == "__main__":
    unittest.main()
