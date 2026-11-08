"""Pure tests for cpu_check's result folding and the export layout. No JAX needed: export runs on synthetic records in a temp dir."""
import json, pathlib, sys, tempfile, unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import common  # noqa: E402
import cpu_check  # noqa: E402
import export  # noqa: E402

REF = "import jax, jax.numpy as jnp\nCONFIG = {'name': 't'}\ndef create_inputs():\n    return (jnp.ones((8, 128)),)\ndef workload(x):\n    return x * 2.0\n"
GOOD = {c: True for c in cpu_check.CHECKS}


def prob(pid, source="kernelbench", status="ok", cc=GOOD, family=None, sem=None):
    code = REF.replace("'t'", repr(pid))
    r = {"problem_id": pid, "family_id": family or pid, "task_name": common.task_name(pid), "source": source,
         "source_ref": {"license": "MIT", **({"semantics_check": sem} if sem is not None else {})}, "ref_code": code, "ref_sha256": common.sha256(code),
         "signature": {"op_chain": ["mul"], "input_shapes": [[8, 128]], "input_dtypes": ["float32"], "output_shapes": [[8, 128]]},
         "input_bytes": 4096, "status": status, "exclude_reason": None if status == "ok" else "contamination:x", "plan": "doubles x"}
    if cc is not None:
        r["cpu_check"] = {**cc, "passed": all(cc.get(c) for c in cpu_check.CHECKS),
                          "failed_checks": [c for c in cpu_check.CHECKS if not cc.get(c)], "jax": "0.10.2", "oracle_sha": "abc"}
    return r


class Fold(unittest.TestCase):
    def test_all_checks_true_passes(self):
        r = cpu_check.fold(dict(GOOD), 0, "", 1.2)
        self.assertTrue(r["passed"]); self.assertEqual(r["failed_checks"], [])

    def test_missing_check_fails_and_is_named(self):
        r = cpu_check.fold({**GOOD, "seeds_distinct": False}, 0, "", 1.0)
        self.assertFalse(r["passed"]); self.assertEqual(r["failed_checks"], ["seeds_distinct"])

    def test_memory_cap_and_timeout_are_reported(self):
        r = cpu_check.fold(None, 137, "...\n[memory cap: 9.1 GB > 8.0 GB]", 9.1)
        self.assertFalse(r["passed"]); self.assertIn("memory cap", r["error"])
        self.assertEqual(cpu_check.fold(None, 124, "", None)["error"], "timeout")


class Export(unittest.TestCase):
    def run_export(self, rows):
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "problems").mkdir()
        common.write_jsonl(d / "problems" / "problems.jsonl", rows)
        with mock.patch.object(export, "PROBLEMS", d / "problems"), mock.patch.object(export, "FINAL", d / "final"):
            m = export.export(0.1)
            return d / "final", m, export.check()

    def test_layout_fields_and_exclusions(self):
        rows = [prob("kb/l1/a"), prob("kb/l1/b", cc={**GOOD, "finite": False}), prob("kb/l1/c", status="excluded"),
                prob("scraped/r/k", source="scraped", sem={"ok": True}), prob("scraped/r/bad", source="scraped", sem={"ok": False}),
                prob("kb/l1/d", cc=None)]
        final, m, bad = self.run_export(rows)
        self.assertEqual(bad, [])
        recs = common.read_jsonl(final / "problems.jsonl")
        self.assertEqual(sorted(r["problem_id"] for r in recs), ["kb/l1/a", "scraped/r/k"])
        for r in recs:
            self.assertEqual(common.sha256((final / r["path"]).read_text()), r["ref_sha256"])
            self.assertTrue(r["path"].endswith(f"{r['task_name']}/baseline.py"))
        reasons = {e["problem_id"]: e["reason"] for e in common.read_jsonl(final / "excluded.jsonl")}
        self.assertIn("finite", reasons["kb/l1/b"]); self.assertIn("contamination", reasons["kb/l1/c"])
        self.assertIn("semantics", reasons["scraped/r/bad"]); self.assertIn("not run", reasons["kb/l1/d"])
        self.assertEqual(m["n_problems"], 2)

    def test_family_never_straddles_split_and_check_catches_tampering(self):
        rows = [prob(f"kb/l2/v{i}", family="kb/l2/base") for i in range(5)]
        final, _, bad = self.run_export(rows)
        self.assertEqual(bad, [])
        self.assertEqual(len({r["split"] for r in common.read_jsonl(final / "problems.jsonl")}), 1)
        p = next((final / "problems").glob("*/baseline.py"))
        p.write_text(p.read_text() + "# edited\n")
        with mock.patch.object(export, "FINAL", final):
            self.assertTrue(any("sha" in b for b in export.check()))


if __name__ == "__main__":
    unittest.main()
