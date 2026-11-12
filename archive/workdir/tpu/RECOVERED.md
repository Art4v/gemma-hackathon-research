# TPU fleet tooling, rebuilt Nov 11 after the reboot wiped /private/tmp

| File | Source |
|---|---|
| `ops1/vm8.sh` | `origin/main:oracle/vm.sh` + transcript patches at line 9727 (ACCEL/CPH, TPU_ENV, chipcheck) and line 9770 (gcloud flag order) |
| `ops1/ops_loop.sh` | `origin/karthik/kbk:oracle/tools/ops_loop.sh` + the sed edits from line 10148 (RUNNER, hnaracodes identity, description). Karthik's newer version requires `PROJECT` from the environment |
| `ops1/crashloop.py` | `hruday/kimi-k31:kimi/tpu/crashloop.py`, the final patched version: oracle-order suspects, halt-victim re-run, shard sync, upload retries |
| `ops1/repo` | Detached worktree of gemma-dataset-claude at `748da03`, the oracle commit of every Kimi verdict. Oracle files hash to `c9a07d9298a1` |
| `fleet/a2mvm1/{grab_v6e,vm8,hunt_up,guard}.sh` | Per-VM template. `grab_v6e.sh` from `origin/karthik/kbk`; `hunt_up.sh` is the final fixed template (bootstrap retried, oracle user verified, deleted on failure); `guard.sh` is line 9745 + the v1 sed + the line 10887/14739 patches (lock file, idle counted from creation) |
| `fleet/fleet_run_n.sh` | `hruday/kimi-k31:kimi/tpu/fleet_run_n.sh` with SC set to `~/Projects/gemma-work` |

Not yet exercised on a real VM after the rebuild: the first v7 TPU run is the test.
