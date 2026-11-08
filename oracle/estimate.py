"""ESTIMATE of TPU v6e chip time for the oracle, per task and in total. Nothing is allocated: shapes come from jax.eval_shape, FLOPs and
bytes from XLA's cost analysis of the reference lowered on CPU. Run time is a roofline at v6e peak, compile time a declared constant,
so read the numbers as an order of magnitude, not a measurement. CPU cost analysis counts unfused bytes, so memory-bound tasks with large
intermediates (attention, SSD) come out slower than a fused TPU program would run.

  python oracle/estimate.py [TASK ...] [--assets ../dataset/assets.json] [--json]

Per task:
  ref_only_s     the reference side, once per task: 2 compiles (precision highest and default), then per seed build the inputs, run
                 highest and default, copy both outputs to the host and compare them; plus 3 warm-up and 20 timed default calls.
  kernel_eval_s  one kernel: 1 compile, per seed build the inputs, run, copy to host and run 3 comparisons; plus 23 timed calls,
                 assuming the kernel runs at the reference's default-precision time.
fits_hbm is the oracle's rule (tpu_check.HBM_BUDGET_BYTES >= 3 x input bytes); tasks that do not fit never reach the chip.
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tpu_oracle as o  # noqa: E402

PEAK_BF16_FLOPS = 918e12  # v6e, bf16 MXU peak
PEAK_HBM_BPS = 1.6e12     # v6e HBM bandwidth
HIGHEST_PASSES = 6        # precision "highest" runs an f32 matmul as about 6 bf16 passes
COMPILE_S = 10.0          # declared per XLA or Mosaic compile; not measured
HOST_BPS = 2e9            # one device-to-host copy, or one numpy comparison pass, over an output
TIMED_CALLS = 23          # tpu_check: 3 warm-up + 20 timed


def task_cost(name: str, code: str, seeds: int) -> dict:
    t = o.Task(name, code)  # its load() keeps the task's static_argnums (JAXBench 11p) concrete
    t.load()
    abstract, f = t.abstract, t.jit(t.ref.workload)
    out = f.eval_shape(*abstract)
    ca = f.lower(*abstract).cost_analysis()
    ca = (ca[0] if isinstance(ca, (list, tuple)) else ca) or {}
    in_b, out_b = o.nbytes(abstract), o.nbytes(out)
    flops, moved = float(ca.get("flops", 0.0)), float(ca.get("bytes accessed", in_b + out_b))
    t_def = max(flops / PEAK_BF16_FLOPS, moved / PEAK_HBM_BPS)
    t_high = max(HIGHEST_PASSES * flops / PEAK_BF16_FLOPS, moved / PEAK_HBM_BPS)
    t_in, t_host = in_b / PEAK_HBM_BPS, out_b / HOST_BPS
    ref_only = 2 * COMPILE_S + seeds * (t_in + t_high + t_def + 4 * t_host) + TIMED_CALLS * t_def
    kernel_eval = COMPILE_S + seeds * (t_in + t_def + 4 * t_host) + TIMED_CALLS * t_def
    return {"task": name, "input_bytes": in_b, "output_bytes": out_b, "fits_hbm": 3 * in_b <= o.tc.HBM_BUDGET_BYTES,
            "gflop": flops / 1e9, "run_default_ms": t_def * 1e3, "ref_only_s": ref_only, "kernel_eval_s": kernel_eval}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tasks", nargs="*", help="default: every dev task in --assets")
    ap.add_argument("--assets", default=str(o.ROOT.parent / "dataset/assets.json"))
    ap.add_argument("--seeds", type=int, default=o.N_SEEDS); ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    o.init(cpu_smoke=True)
    refs = json.loads(pathlib.Path(a.assets).read_text())["dev"]
    rows, failed = [], []
    for t in a.tasks or sorted(refs):
        try:
            rows.append(task_cost(t, refs[t], a.seeds))
        except Exception as e:  # a task that cannot be loaded or traced is listed, not fatal
            failed.append({"task_failed": t, "error": type(e).__name__, "msg": str(e)[-200:]})
    fit = [r for r in rows if r["fits_hbm"]]
    total = {"estimate": True, "tasks": len(rows), "fit": len(fit), "failed": [f["task_failed"] for f in failed],
             "ref_only_s": sum(r["ref_only_s"] for r in fit), "one_kernel_per_task_s": sum(r["kernel_eval_s"] for r in fit),
             "assumptions": {"peak_bf16_flops": PEAK_BF16_FLOPS, "peak_hbm_bps": PEAK_HBM_BPS, "compile_s": COMPILE_S,
                             "highest_passes": HIGHEST_PASSES, "host_bps": HOST_BPS, "seeds": a.seeds}}
    if a.json:
        for r in rows + failed:
            print(json.dumps(r))
        print(json.dumps({"total": total}))
        return
    print(f"ESTIMATE (roofline at v6e peak, compile {COMPILE_S:.0f} s each, {a.seeds} seeds)")
    print(f"{'task':58} {'in GB':>7} {'out GB':>7} {'fits':>5} {'GFLOP':>9} {'run ms':>8} {'ref s':>7} {'kernel s':>8}")
    for r in rows:
        print(f"{r['task']:58} {r['input_bytes'] / 1e9:7.3f} {r['output_bytes'] / 1e9:7.3f} {'yes' if r['fits_hbm'] else 'NO':>5} "
              f"{r['gflop']:9.1f} {r['run_default_ms']:8.2f} {r['ref_only_s']:7.1f} {r['kernel_eval_s']:8.1f}")
    for f in failed:
        print(f"{f['task_failed']:58} could not estimate: {f['error']}: {f['msg']}")
    print(f"total over {total['fit']} of {total['tasks']} tasks that fit: reference-only {total['ref_only_s'] / 60:.1f} min, "
          f"one kernel per task {total['one_kernel_per_task_s'] / 60:.1f} min (ESTIMATE)")


if __name__ == "__main__":
    main()
