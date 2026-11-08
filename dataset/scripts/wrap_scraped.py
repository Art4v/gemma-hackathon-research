"""Stage 2c: wrap public human-written Pallas TPU kernels as problems (plain-JAX reference) plus a human candidate kernel.

  python dataset/scripts/wrap_scraped.py [--limit 5] [--batch pilot] [--backend claude_cli] [--model M] [--list]

Enumerates raw/scrape/**/*.py containing `pallas_call`, drops files that are exam human kernels or their sources (DENY and every
path cited in benchmarks/jaxbench/level2/*/baseline.py headers), and asks the constructor for a reference and a self-contained
wrapper. The reference goes to problems/scraped/<repo>/<name>.py and problems.jsonl; the human kernel becomes a candidate record in
candidates/<batch>/r1-human.jsonl with id <task>.human.s0, so round 1 submits it as a reference-validity probe and, if it passes,
a training target (meta.target_origin human). Scraped kernels are checked by the oracle only; nothing here executes them.
"""
from __future__ import annotations

import argparse, json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import BENCH, CANDIDATES, HBM_BUDGET_BYTES, PIPELINE_VERSION, PROBLEMS, RAW, Isolated, candidate_id, now, read_json, read_jsonl, sha256, task_name, validate, write_jsonl  # noqa: E402
from constructor import Request, get_constructor, save_raw  # noqa: E402
from constructor.prompts import PROMPT_VERSION, WRAP_SCHEMA, WRAP_SYSTEM, wrap_user  # noqa: E402
from author_problems import shape_check  # noqa: E402
from render_chat import unfence  # noqa: E402

REPOS = {"jax": ("jax-ml/jax", "Apache-2.0"), "maxtext": ("AI-Hypercomputer/maxtext", "Apache-2.0"), "tpu-inference": ("vllm-project/tpu-inference", "Apache-2.0")}
# only the TPU kernel directories; never the Pallas implementation, GPU ops, model layers or tests
ROOTS = {"jax": ("jax/experimental/pallas/ops/tpu",), "maxtext": ("src/maxtext/kernels", "src/MaxText/kernels", "MaxText/kernels"),
         "tpu-inference": ("tpu_inference/kernels",)}
# exam human kernels and their upstream families (benchmarks/jaxbench/level2, 1p-11p and 51p-65p): never a problem source
DENY = re.compile(r"(flash_attention|paged_attention|ragged_paged|\brpa\b|_rpa|megablox|gmm|splash|/matmul\.py|csa_streamindex|streamindex|"
                  r"deepseek|\bmla\b|/mla/|gdn|causal_conv1d|mhc/|ragged_gather|ragged_sort|sparsecore|_sc\.py|all_gather|example_kernel|"
                  r"tuned_params|mask_info|schedule\.py|write_kv|kv_transfer|test_|/tests?/|__init__|/random/|ring_attention|ulysses|"
                  r"usp_attention|relayout|backend\.py|/common\.py|util\.py|api\.py|ops\.py|/experimental/)", re.I)


def cited_in_level2() -> set:
    """Basenames of source files the exam's level2 kernels cite in their headers."""
    names = set()
    for p in (BENCH / "level2").glob("*/baseline.py"):
        head = "\n".join(p.read_text().splitlines()[:40])
        names.update(m.lower() for m in re.findall(r"([A-Za-z0-9_]+\.py)", head))
    return names


def enumerate_kernels() -> list:
    cited = cited_in_level2()
    out, seen = [], set()
    for repo in REPOS:
        root = RAW / "scrape" / repo
        for sub in ROOTS[repo]:
            for p in sorted((root / sub).rglob("*.py")) if (root / sub).exists() else []:
                rel = str(p.relative_to(root))
                if "pallas_call" not in p.read_text(errors="replace"):
                    continue
                key = (repo, rel.lower())  # the MaxText clone exposes src/maxtext and src/MaxText on a case-insensitive filesystem
                if DENY.search("/" + rel) or p.name.lower() in cited or key in seen:
                    continue
                seen.add(key)
                out.append((repo, rel, p))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=5); ap.add_argument("--batch", default="pilot"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--backend", default=None, help="claude_cli | anthropic_batch | gemini (default: $CONSTRUCTOR_BACKEND, else claude_cli)"); ap.add_argument("--model", default=None); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    kernels = enumerate_kernels()
    done = {r["problem_id"] for r in read_jsonl(PROBLEMS / "problems.jsonl")} | {r["problem_id"] for r in read_jsonl(PROBLEMS / "excluded.jsonl")}
    kernels = [(repo, rel, p) for repo, rel, p in kernels if f"scraped/{repo}/{pathlib.Path(rel).stem}" not in done]
    print(f"{len(kernels)} candidate kernel files after the denylist")
    if a.list:
        for repo, rel, _ in kernels:
            print(f"  {repo}: {rel}")
        return
    kernels = kernels[:a.limit] if a.limit else kernels
    sources = read_json(RAW / "SOURCES.json")
    con = get_constructor(a.backend, a.model, workers=a.workers)
    raw_dir = CANDIDATES / "raw_responses"
    reqs = [Request(f"wrap/{repo}/{rel}", WRAP_SYSTEM, wrap_user(REPOS[repo][0], rel, p.read_text(errors="replace"), REPOS[repo][1]), WRAP_SCHEMA)
            for repo, rel, p in kernels]
    meta = {f"wrap/{repo}/{rel}": (repo, rel, p) for repo, rel, p in kernels}
    kept, excluded, humans = [], [], []
    for resp in con.generate(reqs):
        repo, rel, p = meta[resp.key]
        save_raw(raw_dir, resp)
        name = pathlib.Path(rel).stem
        pid = f"scraped/{repo}/{name}"
        base = {"problem_id": pid, "family_id": pid, "task_name": task_name(pid), "source": "scraped",
                "source_ref": {"repo": REPOS[repo][0], "commit": sources[repo]["commit"], "path": rel, "license": REPOS[repo][1]},
                "signature": None, "port_check": None, "pipeline_version": PIPELINE_VERSION, "prompt_version": PROMPT_VERSION, "created_at": now()}
        if not resp.ok or resp.output["skip"]:
            reason = f"wrap_skip:{resp.output['reason'][:200]}" if resp.ok else f"wrap_failed:constructor:{resp.error}"
            excluded.append({**base, "ref_code": "", "ref_sha256": sha256(""), "input_bytes": None, "fits_hbm": None, "plan": "", "status": "excluded",
                             "exclude_reason": reason}); print(f"  {pid}: {reason[:90]}"); continue
        ref = (unfence(resp.output["ref_code"]) or resp.output["ref_code"]).strip("\n") + "\n"
        ker = (unfence(resp.output["kernel_code"]) or resp.output["kernel_code"]).strip("\n") + "\n"
        chk = shape_check(ref, a.python)
        if not chk.get("ok"):
            excluded.append({**base, "ref_code": ref, "ref_sha256": sha256(ref), "input_bytes": chk.get("input_bytes"), "fits_hbm": None, "plan": resp.output["plan"],
                             "status": "excluded", "exclude_reason": f"wrap_ref_check:{chk.get('reason')}"[:220]}); print(f"  {pid}: ref check failed {chk.get('reason', '')[:80]}"); continue
        rec = {**base, "ref_code": ref, "ref_sha256": sha256(ref), "input_bytes": chk["input_bytes"], "fits_hbm": 3 * chk["input_bytes"] <= HBM_BUDGET_BYTES,
               "plan": resp.output["plan"], "status": "ok", "exclude_reason": None}
        kept.append(rec)
        out = PROBLEMS / "scraped" / repo / f"{name}.py"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ref); (PROBLEMS / "scraped" / repo / f"{name}.kernel.py").write_text(ker)
        humans.append({"candidate_id": f"{rec['task_name']}.human.s0", "problem_id": pid, "task_name": rec["task_name"], "attempt": 1, "sample": 0,
                       "parent_candidate_id": None, "feedback_in": None, "feedback_in_verdict": None, "prescreen_hint": None,
                       "constructor": {"backend": "human", "model": REPOS[repo][0], "prompt_version": PROMPT_VERSION, "docs_sha256": None},
                       "plan": "human-written kernel from " + rel, "code": ker, "code_sha256": sha256(ker), "usage": {}, "created_at": now()})
        print(f"  {pid}: ok {chk['input_bytes'] / 1e6:.0f} MB")
    for r in kept:
        errs = validate("problem", r)
        if errs:
            raise SystemExit(f"{r['problem_id']}: {errs}")
    for c in humans:
        errs = validate("candidate", c)
        if errs:
            raise SystemExit(f"{c['candidate_id']}: {errs}")
    write_jsonl(PROBLEMS / "problems.jsonl", kept, append=True)
    write_jsonl(PROBLEMS / "excluded.jsonl", excluded, append=True)
    write_jsonl(CANDIDATES / a.batch / "r1-human.jsonl", humans, append=True)
    print(f"wrapped {len(kept)}, excluded {len(excluded)}; human candidates -> candidates/{a.batch}/r1-human.jsonl")


if __name__ == "__main__":
    main()
