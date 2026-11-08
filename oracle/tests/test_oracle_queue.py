"""oracle/make_pack.py, oracle/jobq.py and oracle/collect.py: self-contained job directories for a TPU VM, the request queue, and the
check on returned results. Pack and collect tests are pure. The run.sh tests run the packed oracle on CPU (--cpu-smoke, Pallas interpret
mode) on test_tpu_oracle.py's small hand-written kernels only."""
import hashlib, json, os, pathlib, shlex, shutil, subprocess, sys, tempfile, unittest

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE), str(HERE / "tests")]
from test_tpu_oracle import KERNELS, PY, REF  # noqa: E402

import collect, make_pack  # noqa: E402

REF2 = REF.replace("(16, 128)", "(8, 128)")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()


def write_items(d: pathlib.Path, rows: list) -> pathlib.Path:
    p = d / "items.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def three_items(d: pathlib.Path) -> pathlib.Path:
    """Both input forms: inline code, and paths relative to the items file."""
    (d / "refs").mkdir(); (d / "k").mkdir()
    (d / "refs/t2.py").write_text(REF2); (d / "k/b.py").write_text(KERNELS["plain_jnp"])
    return write_items(d, [{"id": "a", "task": "t", "ref_code": REF, "code": KERNELS["good"]},
                           {"id": "b", "task": "t2", "ref_path": "refs/t2.py", "code_path": "k/b.py"},
                           {"id": "c", "task": "t", "ref_code": REF, "code": KERNELS["raises"]}])


def build(d: pathlib.Path, items_path, job_id="j1", **kw) -> pathlib.Path:
    out = d / "jobs" / job_id
    make_pack.build(make_pack.load_items(items_path), out, job_id=job_id,
                    owner="alice", **{"step": "s1", "seeds": 3, "speed_bar": 0.7, "shards": 1, **kw})
    return out


class Pack(unittest.TestCase):
    def test_manifest_hashes_match_the_files_and_items(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            job = build(d, three_items(d), shards=2)
            m = json.loads((job / "MANIFEST.json").read_text())
            for k, v in {"job_id": "j1", "owner": "alice", "step": "s1", "seeds": [0, 1, 2], "speed_bar": 0.7, "n_items": 3,
                         "shards": 2}.items():
                self.assertEqual(m[k], v, k)
            self.assertIn("oracle_commit", m)
            on_disk = {str(p.relative_to(job)) for p in job.rglob("*") if p.is_file()} - {"MANIFEST.json"}
            self.assertEqual(set(m["files"]), on_disk)
            self.assertTrue({"run.sh", "pack.json", *make_pack.ORACLE_FILES} <= on_disk)
            self.assertEqual(make_pack.ORACLE_FILES, ("tpu_oracle.py", "tpu_check.py", "outside_ops.py"))
            for rel, h in m["files"].items():
                self.assertEqual(hashlib.sha256((job / rel).read_bytes()).hexdigest(), h, rel)
            for rel in make_pack.ORACLE_FILES:  # verbatim
                self.assertEqual((job / rel).read_bytes(), (HERE / rel).read_bytes(), rel)
            src = b"".join((HERE / p).read_bytes() for p in make_pack.ORACLE_FILES)
            self.assertEqual(m["oracle_sha"], hashlib.sha256(src).hexdigest()[:12])  # what tpu_oracle._provenance computes
            self.assertEqual({i["id"]: (i["task"], i["code_sha256"], i["ref_sha256"]) for i in m["items"]},
                             {"a": ("t", sha(KERNELS["good"]), sha(REF)), "b": ("t2", sha(KERNELS["plain_jnp"]), sha(REF2)),
                              "c": ("t", sha(KERNELS["raises"]), sha(REF))})
            pack = json.loads((job / "pack.json").read_text())  # the oracle's input format
            self.assertEqual((set(pack), pack["refs"]), ({"refs", "items"}, {"t": REF, "t2": REF2}))
            self.assertEqual([(i["id"], i["code"], i["ref_sha256"]) for i in pack["items"]],
                             [("a", KERNELS["good"], sha(REF)), ("b", KERNELS["plain_jnp"], sha(REF2)), ("c", KERNELS["raises"], sha(REF))])

    def test_bad_items_are_refused(self):
        good = {"id": "a", "task": "t", "ref_code": REF, "code": "x"}
        cases = {"duplicate id": [good, good], "two references for one task": [good, {**good, "id": "b", "ref_code": REF2}],
                 "no code": [{"id": "a", "task": "t", "ref_code": REF}], "no reference": [{"id": "a", "task": "t", "code": "x"}],
                 "both forms": [{**good, "ref_path": "r.py"}]}
        with tempfile.TemporaryDirectory() as d:
            for name, rows in cases.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    make_pack.load_items(write_items(pathlib.Path(d), rows))

    def test_run_sh_header_names_the_oracle_sha_to_pin(self):
        # Lebron James 6705338278: build_rows --oracle-sha must be the judging oracle's sha, so every pack states it up front
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            out = build(d, three_items(d))
            sha = json.loads((out / "MANIFEST.json").read_text())["oracle_sha"]
            self.assertIn(f"oracle_sha {sha}", "\n".join((out / "run.sh").read_text().splitlines()[:4]))

    def test_run_sh_refuses_a_tpu_run_when_the_metadata_token_is_reachable(self):
        # 2026-11-07: the first sessions' VMs exposed an editor-role token to model-written code; a job must not start on such a VM
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            out = build(d, three_items(d))
            fake = d / "bin"; fake.mkdir()
            (fake / "curl").write_text("#!/bin/sh\nprintf 200\n"); (fake / "curl").chmod(0o755)
            p = subprocess.run(["bash", "run.sh"], cwd=out, capture_output=True, text=True,
                               env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "PYTHON": PY})
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("metadata token is reachable", p.stdout + p.stderr)
            self.assertFalse((out / "shards").exists())

    def test_run_sh_refuses_a_tpu_run_without_curl_to_check_the_block(self):
        # Lebron James 6705406624: a missing curl gives an empty status, which must not count as "blocked"
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            out = build(d, three_items(d))
            bare = d / "bare"; bare.mkdir()
            for tool in ("dirname",):
                (bare / tool).symlink_to(shutil.which(tool))
            p = subprocess.run(["/bin/bash", "run.sh"], cwd=out, capture_output=True, text=True,
                               env={"PATH": str(bare), "PYTHON": PY or sys.executable, "HOME": str(d)})  # any Python: the hash check runs first
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("no curl to verify the metadata block", p.stdout + p.stderr)
            self.assertFalse((out / "shards").exists())

    def _run_with_fakes(self, d, out, tools, extra_env=None):
        fake = d / "fakebin"; fake.mkdir(exist_ok=True)
        # a python that refuses "-m pip": if a check regresses, run.sh's JAX pin must not install anything on this machine
        py = PY or sys.executable
        tools = {"curl": "printf 000", "py": f'[ "$1" = -m ] && [ "$2" = pip ] && exit 97; exec {shlex.quote(py)} "$@"', **tools}  # the path may hold quotes
        for name, body in tools.items():
            (fake / name).write_text(f"#!/bin/sh\n{body}\n"); (fake / name).chmod(0o755)
        return subprocess.run(["bash", "run.sh"], cwd=out, capture_output=True, text=True,
                              env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "PYTHON": str(fake / "py"), **(extra_env or {})})

    def test_run_sh_refuses_a_job_user_with_passwordless_sudo(self):
        # 2026-11-07, session 2: the default VM user has passwordless sudo, so model-written code could lift the metadata block
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); out = build(d, three_items(d))
            p = self._run_with_fakes(d, out, {"sudo": "exit 0"})
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("passwordless sudo", p.stderr)
            self.assertFalse((out / "shards").exists())

    def test_run_sh_refuses_a_job_user_in_an_admin_group(self):
        # Lebron James 6708367610: docker (or sudo, adm, google-sudoers) membership is root by another route
        for group in ("docker", "google-sudoers", "lxd", "wheel", "disk", "root"):
            with tempfile.TemporaryDirectory() as d:
                d = pathlib.Path(d); out = build(d, three_items(d))
                p = self._run_with_fakes(d, out, {"sudo": "exit 1", "id": f"echo oracle {group}"})
                self.assertNotEqual(p.returncode, 0, group)
                self.assertIn("admin group", p.stderr, group)
                self.assertFalse((out / "shards").exists())

    def test_run_sh_refuses_a_writable_container_socket(self):
        # Lebron James 6710229198: docker or containerd sockets at any of the usual paths are root by another route
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); out = build(d, three_items(d))
            sock = d / "containerd.sock"; sock.write_text("")
            fake = d / "fakebin"
            p = self._run_with_fakes(d, out, {"sudo": "exit 1", "id": "echo oracle"}, extra_env={"RUNSH_SOCKETS": f"/nonexistent {sock}"})
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("container socket", p.stderr)
            self.assertFalse((out / "shards").exists())

    def test_on_colab_the_vm_checks_are_skipped_and_said_but_the_file_check_still_holds(self):
        # Colab runs as root and holds no credentials of ours (approved Nov 8): ORACLE_ON_COLAB=1 skips only the Cloud VM checks
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); out = build(d, three_items(d))
            py = PY or sys.executable
            stub = {"curl": "printf 200", "sudo": "exit 0", "id": "echo root",  # every VM check would refuse
                    "py": f'case "$1" in -c|-) exec {shlex.quote(py)} "$@";; *) exit 0;; esac'}  # hash check and pin run; the oracle is a no-op
            p = self._run_with_fakes(d, out, stub, extra_env={"ORACLE_ON_COLAB": "1"})
            self.assertNotIn("refusing to run", p.stderr)
            self.assertIn("VM security checks skipped", p.stdout + p.stderr)
            self.assertTrue((out / "shards").exists())
            (out / "pack.json").write_text("{}")
            p = self._run_with_fakes(d, out, stub, extra_env={"ORACLE_ON_COLAB": "1"})
            self.assertIn("changed or missing since the pack was built", p.stdout + p.stderr)

    def test_run_sh_socket_override_cannot_empty_the_list(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d); out = build(d, three_items(d))
            self.assertIn("/run/containerd/containerd.sock", (out / "run.sh").read_text())
            self.assertIn('${RUNSH_SOCKETS:-', (out / "run.sh").read_text())

    def test_a_numeric_step_from_a_queue_request_builds(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            out = build(d, three_items(d), step=4)
            self.assertIn("(alice, 4)", (out / "run.sh").read_text())
            self.assertEqual(json.loads((out / "MANIFEST.json").read_text())["step"], 4)

    def test_bad_job_id_and_existing_dir_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            items = make_pack.load_items(three_items(d))
            with self.assertRaises(ValueError):
                make_pack.build(items, d / "x", job_id="../evil", owner="a", step="s", seeds=3, speed_bar=0.7, shards=1)
            build(d, d / "items.jsonl")
            with self.assertRaises(FileExistsError):
                build(d, d / "items.jsonl")



def run_sh(job: pathlib.Path) -> subprocess.CompletedProcess:
    p = subprocess.run(["bash", str(job / "run.sh"), "--cpu-smoke"], capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHON": PY, "JAX_PLATFORMS": "cpu"})
    return p


def rows_of(path: pathlib.Path) -> list:
    return [r for r in map(json.loads, path.read_text().splitlines()) if not r.get("meta")]


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class RunSh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(cls.tmp.name)
        items = write_items(d, [{"id": n, "task": "t", "ref_code": REF, "code": KERNELS[n]} for n in ("good", "plain_jnp", "raises")])
        cls.job = build(d, items, shards=2)
        cls.proc = run_sh(cls.job)
        cls.m = json.loads((cls.job / "MANIFEST.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runs_every_shard_on_cpu_and_writes_one_results_file(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout[-1500:] + self.proc.stderr[-1500:])
        rows = {r["id"]: r for r in rows_of(self.job / "results/j1.jsonl")}
        self.assertEqual(len(rows_of(self.job / "results/j1.jsonl")), 3)
        self.assertIn(rows["good"]["verdict"], ("pass", "correct_slow"), rows["good"])
        self.assertTrue(rows["good"]["correct"])
        self.assertEqual((rows["plain_jnp"]["verdict"], rows["raises"]["verdict"]), ("not_a_kernel", "compile"))
        for r in rows.values():  # the job directory alone reproduces the manifest's oracle
            self.assertEqual((r["oracle_sha"], r["oracle_commit"], r["seeds"]), (self.m["oracle_sha"], self.m["oracle_commit"], [0, 1, 2]))
            self.assertEqual(r["ref_sha256"], sha(REF))
        self.assertEqual(sorted(p.name for p in (self.job / "shards").iterdir()), ["0of2.jsonl", "1of2.jsonl"])
        s = collect.check(self.job)
        self.assertTrue(s["valid"], s)

    def test_resume_skips_done_ids(self):
        with tempfile.TemporaryDirectory() as d:
            job = pathlib.Path(shutil.copytree(self.job, pathlib.Path(d) / "j1"))
            shards = {p: p.read_text().splitlines() for p in (job / "shards").glob("*.jsonl")}
            for p, lines in shards.items():  # drop "raises"; mark "good" so a rerun of it would show
                keep = [l for l in lines if '"id": "raises"' not in l]
                keep = [l.replace('"id": "good",', '"id": "good", "sentinel": 1,') for l in keep]
                p.write_text("".join(l + "\n" for l in keep))
            p = run_sh(job)
            self.assertEqual(p.returncode, 0, p.stderr[-1500:])
            rows = rows_of(job / "results/j1.jsonl")
            self.assertEqual(sorted(r["id"] for r in rows), ["good", "plain_jnp", "raises"])
            self.assertEqual(next(r for r in rows if r["id"] == "good").get("sentinel"), 1)

    def test_a_changed_file_stops_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            job = pathlib.Path(shutil.copytree(self.job, pathlib.Path(d) / "j1"))
            (job / "pack.json").write_text((job / "pack.json").read_text().replace("2.0", "3.0"))
            p = run_sh(job)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("pack.json", p.stdout + p.stderr)


@unittest.skipUnless(PY, "needs a JAX interpreter (set SERA_JAX_PY)")
class Queue(unittest.TestCase):
    def test_estimate_refuses_an_over_budget_job_and_build_skips_it(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            q = d / "queue"; q.mkdir()
            three_items(q)
            base = {"owner": "alice", "step": "s1", "items_path": "items.jsonl", "seeds": 3, "speed_bar": 0.7}
            (q / "small.json").write_text(json.dumps({**base, "job_id": "small", "priority": 2, "max_chip_minutes": 60}))
            # the roofline estimate's declared compiles alone (2 for the reference, 1 per kernel, 10 s each) exceed 0.01 min
            (q / "over.json").write_text(json.dumps({**base, "job_id": "over", "priority": 1, "max_chip_minutes": 0.01}))
            jq = lambda *a: subprocess.run([PY, str(HERE / "jobq.py"), *a, "--queue", str(q), "--jobs-dir", str(d / "jobs")],
                                           capture_output=True, text=True, timeout=300, env={**os.environ, "JAX_PLATFORMS": "cpu"})
            p = jq("estimate")
            self.assertEqual(p.returncode, 1, p.stderr[-800:])
            lines = {l.split()[0]: l for l in p.stdout.splitlines() if l.split()}
            self.assertIn("REFUSED", lines["over"])
            self.assertIn("OK", lines["small"])
            p = jq("build", "--shards", "2")
            self.assertEqual(p.returncode, 1, p.stderr[-800:])
            self.assertTrue((d / "jobs/small/MANIFEST.json").exists(), p.stdout + p.stderr)
            self.assertFalse((d / "jobs/over").exists())
            m = json.loads((d / "jobs/small/MANIFEST.json").read_text())
            self.assertEqual((m["shards"], m["priority"], m["max_chip_minutes"]), (2, 2, 60))
            self.assertLess(m["estimate_chip_minutes"], 60)
            p = jq("list")
            self.assertEqual(p.returncode, 0, p.stderr[-800:])
            ids = [l.split()[0] for l in p.stdout.splitlines()[1:]]
            self.assertEqual(ids, ["over", "small"])  # priority 1 before 2
            self.assertIn("built", p.stdout.splitlines()[2])


def result_row(i: dict, m: dict, **kw) -> dict:
    return {"id": i["id"], "task": i["task"], "oracle_sha": m["oracle_sha"], "seeds": m["seeds"], "verdict": "pass", "correct": True,
            "speed_vs_xla": 1.0, **kw}


class Collect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(self.tmp.name)
        self.job = build(d, three_items(d))
        self.m = json.loads((self.job / "MANIFEST.json").read_text())
        (self.job / "results").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        meta = {"meta": True, "oracle": True, "oracle_sha": self.m["oracle_sha"], "seeds": self.m["seeds"]}
        (self.job / "results/j1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [meta, *rows]))

    def test_catches_missing_duplicate_and_oracle_sha_mismatch(self):
        a, b, c = self.m["items"]
        self.write([result_row(a, self.m), result_row(a, self.m), result_row(b, self.m, oracle_sha="000000000000"),
                    result_row(b, self.m, id="zz", seeds=[0])])
        s = collect.check(self.job)
        self.assertFalse(s["valid"])
        self.assertEqual((s["missing"], s["duplicate"], s["unexpected"]), (["c"], ["a"], ["zz"]))
        self.assertEqual((s["oracle_sha_mismatch"], s["seeds_mismatch"]), (["b"], ["zz"]))
        p = subprocess.run([sys.executable, str(HERE / "collect.py"), str(self.job)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn("missing", p.stdout)
        self.assertEqual(json.loads((self.job / "summary.json").read_text())["missing"], ["c"])

    def test_valid_results_summary_counts_verdicts_and_median_speed_of_correct_kernels(self):
        a, b, c = self.m["items"]
        self.write([result_row(a, self.m, speed_vs_xla=2.0), result_row(b, self.m, verdict="correct_slow", speed_vs_xla=0.5),
                    result_row(c, self.m, verdict="wrong", correct=False, speed_vs_xla=9.0)])
        s = collect.check(self.job)
        self.assertTrue(s["valid"], s)
        self.assertEqual(s["verdicts"], {"correct_slow": 1, "pass": 1, "wrong": 1})
        self.assertEqual((s["n_correct"], s["median_speed_vs_xla_correct"]), (2, 1.25))

    def test_a_meta_line_from_another_oracle_is_a_mismatch(self):
        a, b, c = self.m["items"]
        self.write([result_row(i, self.m) for i in (a, b, c)])
        p = self.job / "results/j1.jsonl"
        p.write_text(p.read_text().replace(self.m["oracle_sha"], "ffffffffffff", 1))
        s = collect.check(self.job)
        self.assertFalse(s["valid"])
        self.assertEqual(s["oracle_sha_mismatch"], ["meta"])


if __name__ == "__main__":
    unittest.main()
