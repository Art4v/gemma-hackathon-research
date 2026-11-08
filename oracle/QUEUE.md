# TPU oracle job queue

One operator runs everyone's chip jobs on a Cloud TPU v6e VM. You need only a branch.

## Submit a job

1. Write your items as JSONL, one kernel per line: `{"id", "task", "ref_code", "code"}`, or `{"id", "task", "ref_path", "code_path"}`
   with paths relative to the items file. Ids must be unique. All items of one task must share one reference.
2. Add a request `oracle/queue/<job_id>.json` next to it:

```json
{"job_id": "alice-step3", "owner": "alice", "step": "step3", "items_path": "alice-step3.items.jsonl",
 "seeds": 3, "speed_bar": 0.7, "priority": 2, "max_chip_minutes": 30}
```

   `items_path` is relative to the request file. You can put the items inline as `"items": [...]` instead. Lower priority runs
   first.
3. Check the estimate yourself (JAX on CPU is enough): `python oracle/jobq.py estimate alice-step3`. A job whose estimate is over
   `max_chip_minutes` is refused and never built. It is a roofline: an order of magnitude.
4. Commit both files on your branch, or post them, and tell the operator.

## What the operator does

`jobq.py build` packs each accepted job into `oracle/jobs/<job_id>/`. That directory is self-contained: it holds the oracle,
`pack.json`, `run.sh` and a `MANIFEST.json` with the hashes. The operator copies it to the VM and runs `bash run.sh`.
Back on the Mac, `python oracle/collect.py oracle/jobs/<job_id>` checks that every id is present exactly once, and that the
oracle_sha and the seeds match. It then writes `summary.json`: counts per verdict, and the median speed_vs_xla of the correct kernels.

## VM security

Every job runs model-written code, so that code must not be able to get a credential. Sessions 0 and 1 ran it on VMs with the
default compute service account and the `cloud-platform` scope (roles/editor on the project), and as a user with passwordless
sudo. Session 2 (2026-11-07) used the recipe below, which works:

1. **Create** with the logging and read-only storage scopes, and nothing broader:
   `gcloud compute tpus tpu-vm create NAME --zone=ZONE --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e --scopes=https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/devstorage.read_only`
   - Without `devstorage.read_only`, `tpu-runtime.service` can't pull its container from gcr.io, and JAX fails with
     "Failed to connect to [::]:8353".
   - The TPU API adds `pubsub` on its own.
   - Scopes can't be changed later: delete and recreate.
2. **Job user:** the default user has passwordless sudo, so it could undo any block. Run jobs as a separate user
   (`sudo useradd -m oracle`). It has no sudo, and `/dev/vfio` is world read-write, so it still reaches the TPU.
3. **Block the metadata server** for that user's uid, over IPv4 and IPv6:
   `sudo iptables -I OUTPUT 1 -m owner --uid-owner $(id -u oracle) -d 169.254.169.254 -j REJECT`, plus
   `sudo ip6tables -I OUTPUT 1 -m owner --uid-owner $(id -u oracle) -d fd20:ce::254 -j REJECT`.
4. **TPU topology from env.** libtpu can't ask the blocked metadata server, so run each job as
   `sudo -u oracle -H env TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v6e-1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 bash run.sh`.
   Install JAX as `oracle` from its own home (`cd ~` first; uv otherwise reads `./uv.toml` and fails).
5. **run.sh checks this for itself** before a TPU run, and refuses to start if any check fails:
   - a token request to `169.254.169.254`, `metadata.google.internal` or `[fd20:ce::254]` answers 200;
   - curl is missing;
   - the user has passwordless sudo;
   - the user can write a container socket (`/var/run/docker.sock`, `/run/docker.sock`, `/run/containerd/containerd.sock`);
   - the user is in `docker`, `sudo`, `adm`, `google-sudoers`, `wheel`, `lxd`, `disk` or `root`.
6. **Reading results** in oracle's home (mode 750) needs `sudo` from the default user. That includes stop-condition scripts
   and the results fetch, which must expand its globs as root. Delete the VM when the queue is empty:
   `gcloud compute tpus tpu-vm list --zone=ZONE` must show 0 items.

## Where results appear

The operator commits `results/<job_id>.jsonl` and `summary.json` to `oracle/results/<job_id>/` and replies to your request.
Each row is one oracle verdict (`pass`, `correct_slow`, `wrong`, `not_a_kernel`, `too_large_for_chip` or a failure class) with its
feedback line.
