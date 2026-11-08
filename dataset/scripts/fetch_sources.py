"""Stage 1: shallow-clone the problem sources into dataset/raw/ at pinned commits and record them in raw/SOURCES.json.

  python dataset/scripts/fetch_sources.py [--only kernelbench,jax,maxtext,tpu-inference]

Nothing under raw/ is imported or executed by the pipeline. Licences are recorded (file sha) so each record can cite its source.
"""
import argparse, datetime as dt, json, pathlib, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import RAW, sha256, write_json  # noqa: E402

# name -> (repo url, ref, licence file, paths of interest). `ref` None = default branch HEAD at fetch time (commit recorded).
SOURCES = {
    "kernelbench": ("https://github.com/ScalingIntelligence/KernelBench", None, "LICENSE", ["KernelBench/level1", "KernelBench/level2"]),
    "jax": ("https://github.com/jax-ml/jax", "jax-v0.10.2", "LICENSE", ["jax/experimental/pallas/ops/tpu", "docs/pallas"]),
    "maxtext": ("https://github.com/AI-Hypercomputer/maxtext", None, "LICENSE", ["src/MaxText/kernels", "MaxText/kernels"]),
    "tpu-inference": ("https://github.com/vllm-project/tpu-inference", None, "LICENSE", ["tpu_inference/kernels"]),
}


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def fetch(name: str, dest: pathlib.Path) -> dict:
    url, ref, lic, paths = SOURCES[name]
    if not dest.exists():
        args = ["clone", "--depth", "1", "--quiet"] + (["--branch", ref] if ref else []) + [url, str(dest)]
        git(*args)
    commit = git("rev-parse", "HEAD", cwd=dest)
    lic_path = dest / lic
    present = [p for p in paths if (dest / p).exists()]
    return {"repo": url, "ref": ref, "commit": commit, "licence_file": lic, "licence_sha256": sha256(lic_path.read_bytes()) if lic_path.exists() else None,
            "paths_present": present, "fetched_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", default=",".join(SOURCES))
    a = ap.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / "SOURCES.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in a.only.split(","):
        dest = RAW / ("kernelbench" if name == "kernelbench" else f"scrape/{name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            out[name] = fetch(name, dest)
            print(f"{name}: {out[name]['commit'][:12]} paths {out[name]['paths_present']}")
        except subprocess.CalledProcessError as e:
            out[name] = {"error": e.stderr.strip()[-300:]}
            print(f"{name}: FAILED {e.stderr.strip()[-200:]}", file=sys.stderr)
        write_json(out_path, out)


if __name__ == "__main__":
    main()
