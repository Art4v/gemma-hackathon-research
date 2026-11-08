"""The teacher's vendored Pallas docs (dataset/teacher_docs): pinned upstream bytes, a reproducible build, TPU material only."""
import hashlib, json, pathlib, subprocess, sys, unittest

D = pathlib.Path(__file__).resolve().parents[1]  # pallas_guides/


class TeacherDocs(unittest.TestCase):
    def test_upstream_files_match_sources_at_the_oracle_jax_pin(self):
        src = json.loads((D / "SOURCES.json").read_text())
        self.assertEqual((src["repo"], src["tag"], src["licence"]), ("jax-ml/jax", "jax-v0.10.2", "Apache-2.0"))
        for name, sha in src["files"].items():
            self.assertEqual(hashlib.sha256((D / "upstream" / name).read_bytes()).hexdigest(), sha, name)
        self.assertIn("Apache License", (D / "upstream/LICENSE").read_text())

    def test_committed_docs_are_exactly_what_build_produces(self):
        before = (D / "teacher_docs.md").read_text()
        subprocess.run([sys.executable, str(D / "build.py")], check=True, capture_output=True)
        self.assertEqual((D / "teacher_docs.md").read_text(), before)

    def test_tpu_material_only_and_all_four_sources_present(self):
        text = (D / "teacher_docs.md").read_text()
        self.assertNotIn("# GPU version", text)
        self.assertNotIn("Triton backend", text)
        for p in ("docs/pallas/quickstart.md", "docs/pallas/tpu/details.rst", "docs/pallas/tpu/matmul.md", "docs/pallas/tpu/sparse.md"):
            self.assertIn(f"===== {p} (jax-ml/jax @ jax-v0.10.2) =====", text)
        for needed in ("PrefetchScalarGridSpec", "scratch_shapes", "pl.when", "BlockSpec"):
            self.assertIn(needed, text, needed)


if __name__ == "__main__":
    unittest.main()
