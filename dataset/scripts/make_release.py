"""Build the dataset release bundle from an aarav/dataset-v4 build commit.

  python3 make_release.py COMMIT TAG --max-target-tokens N [--repo CLONE] [--out DIR]

CLONE is a clone of Art4v/gemma-hackathon-research with origin/aarav/dataset-v4, origin/dataset-gen and origin/karthik/{kbk,kbkc,m1}
fetched (default: ./repo next to this script). The KernelBench license is fetched with `gh api` (needs gh).

Writes <out>/<TAG>/ (data files from dataset/v4/final at COMMIT, LICENSES/, ATTRIBUTION.csv, README.md datasheet, SHA256SUMS),
then <out>/<TAG>.tar.gz, <out>/<TAG>.zip and <out>/<TAG>.archives.sha256, and <out>/<TAG>.notes.md for the GitHub release body.
Every count in the datasheet is computed from the files; --max-target-tokens is the Gemma-tokenizer figure from the team's token
check (this script has no tokenizer). Refuses if the MANIFEST hashes do not match the files. Uploading is a separate step.
"""
import argparse, base64, collections, csv, hashlib, json, os, re, shutil, subprocess, tarfile, zipfile

S = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(S, "repo")  # overridden by --repo
DATA = ["train.jsonl", "val.jsonl", "verification.jsonl", "coverage.txt", "MANIFEST.json"]
PROBLEM_SOURCES = [("origin/dataset-gen", "dataset/problems/problems.jsonl"), ("origin/karthik/kbk", "dataset/problems/problems.jsonl"),
                   ("origin/karthik/kbkc", "dataset/problems/problems.jsonl"), ("origin/karthik/m1", "dataset/problems/problems.jsonl")]
KIND_NOTE = {
    "single": "TPU oracle `pass`",
    "repair": "TPU oracle `pass`; earlier turns are real failed attempts with the oracle's feedback",
    "repair_injected": "target TPU `pass`; the failed turn is a verified kernel with an injected error, and its feedback comes from the CPU kernel check",
    "shape_variant": "CPU only; the target kernel passed the TPU oracle at the original shape",
    "generated": "CPU only (template-generated reference and kernel pairs)",
    "correct_slow": "TPU: correct on all seeds but below the 0.7x XLA speed bar",
}


def git(*a, binary=False):
    r = subprocess.run(["git", "-C", REPO, *a], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def attribution(rows, out_path):
    probs = {}
    for ref, path in PROBLEM_SOURCES:
        for line in git("show", f"{ref}:{path}").splitlines():
            if line.strip():
                p = json.loads(line)
                probs.setdefault(p["task_name"], p)
    need = sorted({(r["task"], r["source"]) for r in rows if r["source"] in ("kernelbook", "github", "scraped")})
    out, missing = [], []
    for t, src in need:
        p = probs.get(t) or probs.get(t.split("__m1_")[0])
        if not p:
            missing.append(t)
            continue
        s = p.get("source_ref") or {}
        repo, commit = s.get("repo") or s.get("repo_name") or "", s.get("commit") or s.get("sha") or ""
        link = s.get("repo_link") or s.get("url") or ""
        # Some records keep a GitHub tree URL (".../tree/<sha>") in the repo field; split it into repo, commit and link.
        m = re.match(r"https://github\.com/([^/]+/[^/]+)/tree/([0-9a-f]{40})", repo)
        if m:
            link = link or repo
            repo, commit = m.group(1), commit or m.group(2)
        if not link and repo and commit:
            link = f"https://github.com/{repo}/tree/{commit}"
        out.append({"task": t, "source": src, "repo": repo, "commit": commit,
                    "license": (s.get("license_checked") or s.get("license") or "").lower(), "link": link})
    if missing:
        raise SystemExit(f"no problem record for {len(missing)} attributed tasks, e.g. {missing[:3]}")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "source", "repo", "commit", "license", "link"])
        w.writeheader()
        w.writerows(out)
    return out


def n(x):
    return f"{x:,}"


def datasheet(tag, commit, built_at, train, val, ver, man, attr, max_tok):
    kinds = collections.Counter(r["kind"] for r in train)
    srcs = collections.Counter(r["source"] for r in train)
    vby = collections.Counter(v["verification"] for v in ver)
    tasks = len({r["task"] for r in train})
    overlap = len({r["task"] for r in train} & {r["task"] for r in val})
    dropped = (man.get("dropped_target_over_cap") or {})
    excl = (man.get("excluded_families") or {}).get("families", [])
    kind_rows = "\n".join(f"| `{k}` | {n(kinds[k])} | {KIND_NOTE.get(k, '')} |" for k in sorted(kinds, key=lambda k: -kinds[k]))
    src_txt = ", ".join(f"{n(c)} {s}" for s, c in srcs.most_common())
    lic = collections.Counter(a["license"] or "(none recorded)" for a in attr if a["source"] == "kernelbook")
    parts = ", ".join(f"`{k}` @ `{v}`" for k, v in sorted((man.get("part_commits") or {}).items()))
    return f"""# Pallas TPU kernel dataset, {tag}

Supervised fine-tuning data for teaching a small model (Gemma 4 E4B) to write Pallas TPU kernels from plain-JAX references. Every
kernel used as a training target was judged by a TPU v6e oracle or, for a labelled minority, by a CPU check.

This is the final dataset for the Nov 11, 2026 06:00 UTC data freeze. Built from `aarav/dataset-v4` @ `{commit[:7]}` ({built_at}), whose source
parts are {parts}. Repository: `Art4v/gemma-hackathon-research`.

## Files

| File | What it holds |
|---|---|
| `train.jsonl` | {n(len(train))} training rows (chat format, one row per example) |
| `val.jsonl` | {n(len(val))} validation rows, split by problem family, never trained on ({overlap} tasks shared with train) |
| `verification.jsonl` | one record per training row: how its target was verified, the final oracle verdict, and the oracle job ids |
| `MANIFEST.json` | counts, SHA256 of each data file, the source-part commits, the excluded families and the rows dropped for length |
| `coverage.txt` | structural-skeleton coverage of the training set |
| `ATTRIBUTION.csv` | for every task derived from third-party code: source, repository, pinned commit, license at that commit, link |
| `LICENSES/` | the KernelBook license and notice, and KernelBench's MIT license |
| `SHA256SUMS` | checksums of every file in this bundle |

## Row format

Each row in `train.jsonl` and `val.jsonl` has:
- `example_id`: `<task>.<kind>.<attempt>`
- `task`: the problem name
- `kind`: one of the kinds below
- `source`: where the problem came from
- `teacher`: who wrote the target kernel (`claude`, `injected`, `generator`)
- `thinking`: whether the chat template's thinking flag is set
- `messages`: the conversation, built by the evaluation's own prompt function, so a training row is byte-identical to what the model
  sees at test time

A single-turn row is the problem followed by a passing kernel. A repair row is the problem, a failed kernel, the oracle's feedback
line verbatim, and so on up to the passing kernel.

## Counts (training set)

| Kind | Rows | How the target was verified |
|---|---|---|
{kind_rows}

By verification: {n(vby.get('tpu', 0))} rows TPU `pass`, {n(vby.get('tpu_correct_slow', 0))} TPU `correct_slow`, {n(vby.get('cpu', 0))} CPU only.
By source: {src_txt}. The rows cover {n(tasks)} distinct tasks.

The {n(kinds.get('correct_slow', 0))} `correct_slow` targets were kept by team decision (they came from the `v3` base set). If a variant
without them is published, it will be a separate `dataset-v4.1` release; this tag will not change.

{len(dropped.get('ids', []))} rows were dropped because their target exceeded {n(dropped.get('max_chars', 0))} characters (`MANIFEST.json`,
`dropped_target_over_cap`). On the Gemma 4 E4B tokenizer the longest target in `train.jsonl` is {n(max_tok)} tokens, inside the
evaluation's 8,192-token output limit.

## How kernels were judged

The oracle (`oracle/`, files hashed as `c9a07d9298a1`, unchanged across every row here) runs on a TPU v6e at the task's full input
size and gives the first failing step as the verdict:
1. The solution must be a real Pallas kernel: a `pallas_call` the output depends on, with only layout, dtype and small integer work
   outside it (`not_a_kernel` otherwise).
2. It must compile and run (`compile`, `vmem`, `mosaic`, `timeout`, `runtime`).
3. It must match a float32 answer key on 3 input seeds (`wrong`).
4. It must run at least 0.7x as fast as XLA (`correct_slow` below that), timed as the median of 20 calls after 3 warm-ups.

Results from v6e-1 and v6e-8 hosts were cross-checked: on 22 kernels the verdicts were identical and the median speed ratio between
hosts was 0.997. A kernel within about 10% of the 0.7 bar can change verdict between any two runs.

Every problem passed CPU readiness checks at full size. Five KernelBook problems whose float32 reference is NaN on every oracle seed
(exact zeros in the uniform inputs give 0/0 or log(0)) cannot be passed by any kernel; they are excluded (`problems.jsonl`,
`exclude_reason: nan_reference`).

## Contamination control

Every problem was screened against all 73 JAXBench files (the evaluation benchmark) by KernelBench-stem match, jaxpr op-chain graph
signatures, near-duplicate text on identifier-kept and identifier-abstracted token streams, and an algorithm review. Problems that
share a JAXBench task's core algorithm were excluded, as were these {len(excl)} families and their structural mutations:
{", ".join(f"`{f}`" for f in excl)}.

## Licenses

- **KernelBook-derived rows** (`source: kernelbook`): the PyTorch modules come from GPU MODE's KernelBook (`dataset_permissive.parquet`)
  under the June 9 Researcher Reciprocity License (`LICENSES/KernelBook-June9-Researcher-Reciprocity-License.txt`, unmodified; notice in
  `LICENSES/KernelBook-NOTICE.md`). The JAX references are ports written for this dataset. **Attachment A applies to any model trained
  on these rows:** researchers must be able to generate with, evaluate and publish about it on the same terms as other users. Each
  module also keeps its own repository's license, checked at the pinned commit (`ATTRIBUTION.csv`: {", ".join(f"{k} {v}" for k, v in lic.most_common())}).
- **KernelBench-derived rows** (`source: kernelbench`): ported from KernelBench, MIT license (`LICENSES/KernelBench-MIT.txt`).
- **GitHub and scraped rows**: reference implementations of public kernels; repository, commit and license per task in
  `ATTRIBUTION.csv`.
- **Authored and generated rows**, and all constructor-written kernels: written for this project. The team has not yet chosen a license
  for its own contributions; until it does, treat them as research use only and ask before redistributing.

## Integrity

```
sha256sum -c SHA256SUMS
```

`MANIFEST.json` also records the SHA256 of each data file as written by the build.
""", dict(kinds=kinds, srcs=srcs, vby=vby, tasks=tasks, lic=lic, overlap=overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commit"); ap.add_argument("tag")
    ap.add_argument("--max-target-tokens", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(S, "release"))
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    globals()["REPO"] = os.path.abspath(a.repo)
    commit = git("rev-parse", f"{a.commit}^{{commit}}").strip()
    R = os.path.join(a.out, a.tag)
    if os.path.exists(R):
        raise SystemExit(f"{R} exists; move it away first")
    os.makedirs(os.path.join(R, "LICENSES"))
    for f in DATA:
        with open(os.path.join(R, f), "wb") as fh:
            fh.write(git("show", f"{commit}:dataset/v4/final/{f}", binary=True))
    man = json.load(open(os.path.join(R, "MANIFEST.json")))
    for f, h in (man.get("files") or {}).items():
        if sha(os.path.join(R, f)) != h:
            raise SystemExit(f"MANIFEST hash mismatch for {f}")
    for src, dst in (("dataset/kernelbook/LICENSE.txt", "KernelBook-June9-Researcher-Reciprocity-License.txt"),
                     ("dataset/kernelbook/NOTICE.md", "KernelBook-NOTICE.md")):
        with open(os.path.join(R, "LICENSES", dst), "wb") as fh:
            fh.write(git("show", f"origin/karthik/kbk:{src}", binary=True))
    kb = subprocess.run(["gh", "api", "repos/ScalingIntelligence/KernelBench/license", "--jq", ".content"], capture_output=True, check=True, text=True).stdout
    with open(os.path.join(R, "LICENSES", "KernelBench-MIT.txt"), "wb") as fh:
        fh.write(base64.b64decode(kb))
    train, val, ver = (jl(os.path.join(R, f)) for f in ("train.jsonl", "val.jsonl", "verification.jsonl"))
    if len(train) != man["n_train"] or len(val) != man["n_val"] or len(ver) != len(train):
        raise SystemExit("row counts disagree with MANIFEST")
    attr = attribution(train + val, os.path.join(R, "ATTRIBUTION.csv"))
    built_at = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%cd", "--date=format-local:%b %d, %H:%M UTC", commit],
                              capture_output=True, check=True, text=True, env={**os.environ, "TZ": "UTC"}).stdout.strip()
    readme, st = datasheet(a.tag, commit, built_at, train, val, ver, man, attr, a.max_target_tokens)
    open(os.path.join(R, "README.md"), "w").write(readme)
    files = sorted(os.path.relpath(os.path.join(d, f), R) for d, _, fs in os.walk(R) for f in fs)
    open(os.path.join(R, "SHA256SUMS"), "w").write("".join(f"{sha(os.path.join(R, f))}  {f}\n" for f in files))
    tgz, zp = os.path.join(a.out, f"{a.tag}.tar.gz"), os.path.join(a.out, f"{a.tag}.zip")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(R, arcname=a.tag)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + ["SHA256SUMS"]:
            z.write(os.path.join(R, f), f"{a.tag}/{f}")
    open(os.path.join(a.out, f"{a.tag}.archives.sha256"), "w").write(f"{sha(tgz)}  {a.tag}.tar.gz\n{sha(zp)}  {a.tag}.zip\n")
    k, v = st["kinds"], st["vby"]
    notes = (f"Pallas TPU kernel SFT dataset for Gemma 4 E4B, final for the Nov 11, 2026 06:00 UTC data freeze. Built from `aarav/dataset-v4` @ `{commit[:7]}`.\n\n"
             f"| | |\n|---|---|\n| Train / val rows | {n(len(train))} / {n(len(val))} ({st['overlap']} shared tasks) |\n| Distinct tasks | {n(st['tasks'])} |\n"
             f"| Target verification | {n(v.get('tpu', 0))} TPU `pass`, {n(v.get('tpu_correct_slow', 0))} TPU `correct_slow`, {n(v.get('cpu', 0))} CPU only |\n"
             f"| Kinds | {', '.join(f'{n(c)} {kk}' for kk, c in k.most_common())} |\n"
             f"| Sources | {', '.join(f'{n(c)} {s}' for s, c in st['srcs'].most_common())} |\n"
             f"| Oracle | TPU v6e, files hashed `c9a07d9298a1`, speed bar 0.7x XLA |\n| Longest target | {n(a.max_target_tokens)} tokens (Gemma 4 tokenizer), limit 8,192 |\n\n"
             f"Assets: `{a.tag}.tar.gz` and `{a.tag}.zip` (the same {len(files) + 1} files), and `{a.tag}.archives.sha256`. Inside, `README.md` is the "
             f"datasheet, `SHA256SUMS` covers every file, and `ATTRIBUTION.csv` gives repository, pinned commit and license for all {n(len(attr))} "
             f"third-party-derived tasks.\n\nThe {n(k.get('correct_slow', 0))} `correct_slow` targets (from the `v3` base) are kept by team decision; a variant without them would be a separate `dataset-v4.1` release, and this tag will not change.\n\n**License note:** rows derived from KernelBook carry the June 9 Researcher Reciprocity License, and its "
             f"Attachment A applies to models trained on them (see `LICENSES/`). KernelBench-derived rows are MIT. The team has not yet chosen a "
             f"license for its own authored rows and constructor kernels; treat them as research use only until it does.\n")
    open(os.path.join(a.out, f"{a.tag}.notes.md"), "w").write(notes)
    print(f"{R}: {len(files) + 1} files; train {len(train)} val {len(val)}; attribution {len(attr)}; archives written")


if __name__ == "__main__":
    main()
