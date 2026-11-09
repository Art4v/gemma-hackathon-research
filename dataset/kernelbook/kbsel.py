"""Select KernelBook modules to port: dedup, drop trivial or non-portable ones, spread across op families and repos, verify each repo's
license at the pinned commit (GitHub API), split into shards A/B/C. Static: reads source text only, runs nothing.
  python -I kbsel.py N_PER_SHARD     (reads dataset_permissive.parquet; writes selection.jsonl)
"""
import collections, concurrent.futures as cf, hashlib, json, pathlib, random, re, subprocess, sys

import pyarrow.parquet as pq

D = pathlib.Path(__file__).resolve().parent
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "0bsd", "unlicense", "zlib"}
BAD = re.compile(r"\.item\(\)|\.tolist\(\)|\.numpy\(\)|nonzero|unique\(|torch\.rand|randn|randint|dropout|\.cuda\(|print\(|np\.|import numpy|"
                 r"torchvision|pretrained|load_state_dict|\.to\(device|while |input\(|open\(")
FAMILIES = [("attention", r"softmax.*(matmul|bmm|einsum|@)|(matmul|bmm|einsum|@).*softmax"), ("conv", r"conv\d?d|Conv\dd|unfold"),
            ("norm", r"norm|LayerNorm|GroupNorm|BatchNorm|InstanceNorm|rsqrt|std\(|var\("), ("pool", r"pool|Pool"),
            ("matmul", r"Linear|matmul|bmm|einsum|mm\(|@"), ("loss", r"loss|Loss|mean\(\)|sum\(\)"),
            ("reduction", r"\.sum\(|\.mean\(|\.max\(|\.min\(|logsumexp|cumsum|cumprod|amax|argmax"), ("elementwise", r".")]


def forward_body(code):
    m = re.search(r"def forward\(self[^)]*\):\n((?:[ \t]+.*\n|\n)+)", code)
    return m.group(1) if m else ""


def norm_src(code):
    code = re.sub(r"#.*", "", code)
    code = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", code)
    return re.sub(r"\s+", " ", code).strip()


def family(code):
    body = forward_body(code) + code
    return next(f for f, rx in FAMILIES if re.search(rx, body))


def ok(row):
    code, body = row["python_code"], forward_body(row["python_code"])
    lines = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("#")]
    calls = set(re.findall(r"(?:torch|F|self\.\w+|nn)\.?\w*\(", body))
    if "def get_inputs" not in code or "def get_init_inputs" not in code or not (2 <= len(lines) <= 40):
        return False
    if BAD.search(body) or len(calls) < 2:
        return False
    return len(code) < 6000


CACHE = D / "license_cache.json"
_cache = __import__("json").loads(CACHE.read_text()) if CACHE.exists() else {}


def license_at_commit(row):
    key = f"{row['repo_name']}@{row['sha']}"
    if key in _cache:
        return _cache[key]
    owner_repo = row["repo_name"]
    r = subprocess.run(["gh", "api", f"repos/{owner_repo}/license?ref={row['sha']}", "--jq", ".license.spdx_id"],
                       capture_output=True, text=True, timeout=60)
    _cache[key] = r.stdout.strip().lower() if r.returncode == 0 else None
    return _cache[key]


def main(n_per_shard):
    rows = pq.read_table(D / "dataset_permissive.parquet").to_pylist()
    seen, pool = set(), []
    for r in rows:
        h = hashlib.sha256(norm_src(r["python_code"]).encode()).hexdigest()
        if h in seen or not ok(r):
            continue
        seen.add(h)
        pool.append({**r, "family": family(r["python_code"])})
    rng = random.Random(0)
    rng.shuffle(pool)
    fams = collections.defaultdict(list)
    for r in pool:
        fams[r["family"]].append(r)
    print(f"{len(rows)} rows, {len(pool)} after dedup + portability filter;", {f: len(v) for f, v in fams.items()})
    want = 3 * n_per_shard
    # round-robin across families (so no family dominates), at most 2 modules per repo, oversample 1.3x for license failures
    per_repo, picked = collections.Counter(), []
    while len(picked) < int(want * 1.3) and any(fams.values()):
        for f in list(fams):
            while fams[f]:
                r = fams[f].pop()
                if per_repo[r["repo_name"]] < 2:
                    per_repo[r["repo_name"]] += 1
                    picked.append(r)
                    break
    with cf.ThreadPoolExecutor(8) as ex:
        lics = list(ex.map(license_at_commit, picked))
    CACHE.write_text(json.dumps(_cache))
    kept = [dict(r, license_checked=l) for r, l in zip(picked, lics) if l in PERMISSIVE][:want]
    print(f"license check at the pinned commit: {sum(l in PERMISSIVE for l in lics)} of {len(picked)} permissive; kept {len(kept)}",
          collections.Counter(lics).most_common(6))
    with open(D / "selection.jsonl", "w") as f:
        for i, r in enumerate(kept):
            stem = re.sub(r"[^A-Za-z0-9_]", "_", f"{r['module_name']}_{str(r['uuid'])[:8]}")
            f.write(json.dumps({"uuid": r["uuid"], "stem": stem, "module_name": r["module_name"], "repo_name": r["repo_name"], "sha": r["sha"],
                                "repo_link": r["repo_link"], "license_checked": r["license_checked"], "family": r["family"],
                                "shard": "ABC"[i % 3], "python_code": r["python_code"]}) + "\n")
    print("families kept:", collections.Counter(r["family"] for r in kept))


if __name__ == "__main__":
    assert family("def forward(self, x):\n        return torch.softmax(x @ x.T, -1)\n") == "attention"
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
