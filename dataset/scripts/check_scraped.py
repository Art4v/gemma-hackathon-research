"""Semantics check for scraped problems: does the plain-JAX reference compute what the public Pallas kernel computes?

  python dataset/scripts/check_scraped.py [--delete-kernels] [--max-rss-gb 6]

Why: a scraped problem's reference was written by the constructor agent from the kernel's source. Nothing else checks that it
means the same thing. The kernel is the ground truth for its own semantics, so the two are run side by side.

How, per problem (one isolated `python -I` subprocess on CPU, one at a time, memory-capped):
  1. Load the reference (problems/scraped/<repo>/<name>.py) and the wrapped human kernel (<name>.kernel.py).
  2. Override CONFIG in both modules with SMALL[problem_id] (below): sizes that keep every block-size divisibility the kernel
     needs but fit comfortably in CPU memory. The overrides are part of the recorded result.
  3. Build inputs with the reference's create_inputs() for seeds 0, 1, 2 using the oracle's own seeding (tpu_oracle.make_inputs).
  4. Run the kernel with pl.pallas_call forced into interpret mode (tpu_oracle.interpret(True)); run the reference at default
     precision and the oracle's answer key (float32 inputs, matmul precision "highest").
  5. Pass on a seed if tpu_check.close(answer, kernel) or the oracle's xla_eq rule (kernel error <= 2x the reference's own error vs
     the answer key). The problem passes if every seed passes.
Result goes on the problem record as source_ref.semantics_check. A failing problem is excluded as `ref_suspect`.
--delete-kernels removes the .kernel.py files afterwards; source_ref.repo/commit/path and kernel_sha256 keep the check reproducible
(re-fetch with fetch_sources.py and re-wrap with wrap_scraped.py).
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ORACLE, PROBLEMS, Isolated, now, read_jsonl, sha256, write_jsonl  # noqa: E402

ORACLE_FILES = ("tpu_oracle.py", "tpu_check.py", "outside_ops.py")

# Small CONFIGs for interpret mode. Each keeps the kernel's tiling constraints: block sizes unchanged, dims multiples of them.
SMALL = {
    "scraped/maxtext/ragged_attention": {"batch_size": 2, "num_q_heads": 8, "num_kv_heads": 2, "head_dim": 128, "seq_len": 512, "block_size": 256},
    "scraped/tpu-inference/blockwise_kernel": {"n_batch": 512, "n_in": 1024, "n_out": 1024, "block_size": 128},  # kernel blocks: batch 512, in/out 1024
    "scraped/tpu-inference/spmm": {"M": 256, "K": 1024, "N": 512},
}

CHILD = r'''
import json, sys, importlib.util
sys.path.insert(0, ".")
import tpu_oracle as o
o.init(cpu_smoke=True)
jax, jnp, np, tc = o.jax, o.jnp, o.np, o.tc
small = json.loads(open("small.json").read())
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m)
    if hasattr(m, "CONFIG"):
        m.CONFIG.update(small)
    return m
ref = load("ref.py", "ref"); ker = load("kernel.py", "kernel")
f32 = lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x
seeds, shapes = [], None
for s in (0, 1, 2):
    xs = o.make_inputs(ref, s)
    shapes = shapes or [list(x.shape) for x in jax.tree_util.tree_leaves(xs)]
    xla = o._host(jax.jit(ref.workload)(*xs))
    with jax.default_matmul_precision("highest"):
        want = o._host(jax.jit(ref.workload)(*jax.tree_util.tree_map(f32, xs)))
    with o.interpret(True):
        got = o._host(jax.jit(ker.workload)(*xs))
    ek, ex = tc.rel_err(want, got), tc.rel_err(want, xla)
    ok = bool(tc.close(want, got)) or o.xla_eq(ek, ex)
    seeds.append({"seed": s, "ok": ok, "err_kernel": ek, "err_ref_default": ex})
print(json.dumps({"ok": all(x["ok"] for x in seeds), "seeds": seeds, "shapes": shapes,
                  "max_rel_err": max((x["err_kernel"] for x in seeds if x["err_kernel"] is not None), default=None)}))
'''


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--delete-kernels", action="store_true"); ap.add_argument("--max-rss-gb", type=float, default=6.0)
    ap.add_argument("--timeout", type=int, default=1800); ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    path = PROBLEMS / "problems.jsonl"
    rows = read_jsonl(path)
    files = {p: (ORACLE / p).read_text() for p in ORACLE_FILES}
    for r in rows:
        if r["source"] != "scraped" or r["status"] != "ok":
            continue
        pid = r["problem_id"]
        kpath = PROBLEMS / "scraped" / pid.split("/")[1] / (pid.split("/")[2] + ".kernel.py")
        if not kpath.exists():
            print(f"  {pid}: no kernel file ({kpath.name}); {'already checked' if r['source_ref'].get('semantics_check') else 'cannot check'}")
            continue
        small = SMALL.get(pid)
        if small is None:
            print(f"  {pid}: no SMALL config defined; add one to check_scraped.py"); continue
        kernel = kpath.read_text()
        res = Isolated(a.python, timeout=a.timeout, max_rss_gb=a.max_rss_gb).run(
            ["check.py"], files={**files, "check.py": CHILD, "ref.py": r["ref_code"], "kernel.py": kernel, "small.json": json.dumps(small)})
        line = next((l for l in res.stdout.splitlines() if l.startswith("{")), None)
        chk = json.loads(line) if line else {"ok": False, "error": (res.stderr.strip().splitlines() or ["no output"])[-1][-300:]}
        chk.update(method="pallas_interpret_cpu", small_config=small, kernel_sha256=sha256(kernel), kernel_path_upstream=r["source_ref"].get("path"),
                   peak_rss_gb=getattr(res, "peak_rss_gb", None), checked_at=now())
        r["source_ref"]["semantics_check"] = chk
        if not chk["ok"]:
            r["status"], r["exclude_reason"] = "excluded", f"ref_suspect:{str(chk.get('error') or chk.get('max_rel_err'))[:150]}"
        write_jsonl(path, rows)
        print(f"  {pid}: {'PASS' if chk['ok'] else 'FAIL'} max_rel_err {chk.get('max_rel_err')} {chk.get('error', '')[:160]}", flush=True)
    if a.delete_kernels:
        for k in PROBLEMS.glob("scraped/**/*.kernel.py"):
            k.unlink(); print(f"deleted {k.relative_to(PROBLEMS)}")


if __name__ == "__main__":
    main()
