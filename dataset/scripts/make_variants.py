"""Stage 2d: dtype variants of KernelBench ports. For every ok port whose create_inputs defaults to float32, write a bfloat16 variant:
the same module with the default dtype switched, so the oracle (which calls create_inputs() with no arguments) feeds bf16 inputs and
a kernel must handle bf16 loads, f32 accumulation and the bf16 store. The variant shares the port's family_id, so the split keeps them
together. Nothing is executed here: cpu_check.py and the filters gate every variant like any other problem.

  python dataset/scripts/make_variants.py [--batch NAME] [--dry-run]
"""
from __future__ import annotations

import argparse, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import PIPELINE_VERSION, PROBLEMS, now, read_jsonl, sha256, task_name, validate, write_jsonl  # noqa: E402

DEFAULT_F32 = re.compile(r"def create_inputs\(\s*dtype\s*=\s*jnp\.float32\s*\)")


def variant_code(code: str) -> str | None:
    if len(DEFAULT_F32.findall(code)) != 1:
        return None
    out = DEFAULT_F32.sub("def create_inputs(dtype=jnp.bfloat16)", code)
    return out.replace('"""', '"""[bf16 variant] ', 1) if out.startswith('"""') else "# bf16 variant\n" + out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", default=None, help="batch to assign the variants to (default: decided later by the driver)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows = read_jsonl(PROBLEMS / "problems.jsonl")
    have = {r["problem_id"] for r in rows}
    new = []
    for r in rows:
        if r["source"] != "kernelbench" or r["status"] != "ok" or r["problem_id"].endswith("__bf16") or not (r.get("cpu_check") or {}).get("passed"):
            continue
        pid = r["problem_id"] + "__bf16"
        code = variant_code(r["ref_code"])
        if pid in have or code is None:
            continue
        v = {k: r[k] for k in ("family_id", "source", "port_check", "plan", "prompt_version") if k in r}
        v.update(problem_id=pid, task_name=task_name(pid), source_ref={**r["source_ref"], "variant_of": r["problem_id"], "config_delta": {"dtype": "bfloat16"}},
                 ref_code=code, ref_sha256=sha256(code), signature=None, input_bytes=None, fits_hbm=None, status="ok", exclude_reason=None,
                 pipeline_version=PIPELINE_VERSION, created_at=now(), **({"batch": a.batch} if a.batch else {}))
        errs = validate("problem", v)
        if errs:
            raise SystemExit(f"{pid}: {errs}")
        new.append(v)
    print(f"{len(new)} bf16 variants")
    if a.dry_run:
        return
    for v in new:
        out = PROBLEMS / "kb" / f"l{v['source_ref']['kb_level']}" / f"{v['source_ref']['kb_stem']}__bf16.py"
        out.write_text(v["ref_code"])
    write_jsonl(PROBLEMS / "problems.jsonl", rows + new)


if __name__ == "__main__":
    main()
