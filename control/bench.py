"""The control benchmark: base Gemma writes Pallas kernels for JAXBench, the TPU oracle judges them, failures go round again with the
oracle's feedback. Arms: bare (the task only) and docs (pallas_guides/teacher_docs.md in front of the task). See control/DESCRIPTION.md.

  python control/bench.py tasks                                    # the task list and where each reference comes from
  python control/bench.py generate --arm bare|docs [--tasks T ...] [--limit N]   # the next attempt of every task that has one
  python control/bench.py items [--out ITEMS.jsonl]                # attempts waiting for the oracle, both arms, as oracle items
  python control/bench.py ingest RESULTS.jsonl [...]               # the oracle's rows for those attempts
  python control/bench.py summary

One run is one model. Rows are append-only in control/runs/<model tag>/<arm>.jsonl, so a stop and rerun resumes: an attempt is generated
once and never again. An attempt's oracle item id is "<arm>/<task>/t<turn>". Turn 0 is the first attempt; turns 1 and 2 are retries, each
seeing every earlier attempt with the oracle's feedback line on it. A task stops at a pass, at too_large_for_chip or harness_oom (not the
kernel's fault: retrying the code can't help), or after turn 2.

Prompts are planningGemma's rounds.conversation and rollouts.repair_turn, word for word, so the LoRA can be trained on the same format.
Sampling: Ollama /api/chat, thinking off, temperature 0, seed crc32("<task>/0/<turn>") (the same for both arms), num_ctx 65536 for both
arms, at most MAX_TOKENS output tokens. Every row records the prompt's sha256 (template only), token counts, finish reason and time.
"""
import argparse, collections, hashlib, json, pathlib, re, statistics, sys, time, urllib.request, zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks/jaxbench"
REFS = ROOT / "control/refs"  # plain-JAX references for the level 2-only tasks
DOCS = ROOT / "pallas_guides/teacher_docs.md"
RUNS = ROOT / "control/runs"

MODEL = "gemma4:e4b-it-qat"
OLLAMA = "http://localhost:11434/api/chat"
NUM_CTX, MAX_TOKENS, TURNS = 65536, 8192, 3  # 8192: a reply is 1-2k tokens; a repetition loop at temperature 0 stops sooner
ARMS = ("bare", "docs")
TERMINAL = {"pass", "too_large_for_chip", "harness_oom"}

# planningGemma, cloud-gemma/sera-e4b/jaxbench/rounds.py (Gemini's wording, Nov 7 2026) and rollouts.repair_turn, verbatim
SYSTEM = ("You are a code model that writes TPU kernels in JAX Pallas. Given a reference JAX function, reply with one Python code block only. Define "
          "workload(*inputs) with the same signature and outputs as the reference, reading shapes from inputs at runtime. Run the main computation inside "
          "pl.pallas_call, using the imports from jax.experimental import pallas as pl and from jax.experimental.pallas import tpu as pltpu. The kernel runs on a "
          "TPU v6e at the reference's full size. Use the largest blocks fitting in memory to be as fast as the jax.jit reference.")
BEFORE_REF, AFTER_REF = "Here is the reference JAX function:", "Please convert this function into a Pallas TPU kernel."


def tasks() -> dict:
    """{task: (reference code, source)}: JAXBench level 1's plain-JAX baseline, or for a level 2-only task its reference in control/refs."""
    out = {p.parent.name: (p.read_text(), f"jaxbench/level1/{p.parent.name}") for p in sorted((BENCH / "level1").glob("*/baseline.py"))}
    for p in sorted(REFS.glob("*.py")):
        out.setdefault(p.stem, (p.read_text(), f"control/refs/{p.name}"))
    return dict(sorted(out.items(), key=lambda kv: int(re.match(r"\d+", kv[0]).group())))


def conversation(ref: str, history: list, docs: str | None) -> list:
    head = f"Pallas documentation:\n{docs}\n\n" if docs else ""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{head}{BEFORE_REF}\n```python\n{ref}\n```\n{AFTER_REF}"}]
    for past in history:
        messages += [{"role": "assistant", "content": f"```python\n{past['code']}\n```"},
                     {"role": "user", "content": past["feedback"] + "\nReply with the corrected Python code block."}]
    return messages


def extract_code(text: str) -> str:
    found = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text or "", re.S | re.I)
    return found[-1] if found else ""


def seed_for(task: str, turn: int) -> int:
    return zlib.crc32(f"{task}/0/{turn}".encode()) & 0x7fffffff


def docs_for(arm: str):
    return DOCS.read_text() if arm == "docs" else None


def prompt_sha256(arm: str) -> str:
    return hashlib.sha256(json.dumps(conversation("<REFERENCE>", [], docs_for(arm))).encode()).hexdigest()


def item_id(arm: str, task: str, turn: int) -> str:
    return f"{arm}/{task}/t{turn}"


def run_dir(model: str) -> pathlib.Path:
    return RUNS / re.sub(r"[^A-Za-z0-9._-]", "_", model)


def read_jsonl(p: pathlib.Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def verdicts(model: str) -> dict:
    """{item id: the oracle's row}; the latest ingested row wins."""
    return {r["id"]: r for r in read_jsonl(run_dir(model) / "verdicts.jsonl")}


def pending(arm: str, rows: list, vs: dict, all_tasks: dict) -> list:
    """(task, turn, history) for every task whose next attempt can be generated now. A task waiting for a verdict waits."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    jobs = []
    for task in all_tasks:
        done = sorted(by[task], key=lambda r: r["turn"])
        if not done:
            jobs.append((task, 0, []))
            continue
        last = done[-1]
        v = vs.get(item_id(arm, task, last["turn"]))
        if v is None or v["verdict"] in TERMINAL or last["turn"] + 1 >= TURNS:
            continue
        history = [{"code": r["code"], "feedback": vs[item_id(arm, task, r["turn"])]["feedback"]} for r in done]
        jobs.append((task, last["turn"] + 1, history))
    return jobs


def chat(messages: list, seed: int, url=OLLAMA, model=MODEL, timeout=1800) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": seed, "num_ctx": NUM_CTX, "num_predict": MAX_TOKENS}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def generate(arm: str, model: str, only=(), limit=0, chat_fn=chat, log=print) -> int:
    all_tasks = tasks()
    if only:
        all_tasks = {t: v for t, v in all_tasks.items() if t in only}
    out = run_dir(model) / f"{arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = pending(arm, read_jsonl(out), verdicts(model), all_tasks)[: limit or None]
    docs, sha = docs_for(arm), prompt_sha256(arm)
    for i, (task, turn, history) in enumerate(jobs):
        ref, source = all_tasks[task]
        seed, t0 = seed_for(task, turn), time.time()
        r = chat_fn(conversation(ref, history, docs), seed)
        text = r.get("message", {}).get("content", "")
        row = {"id": item_id(arm, task, turn), "task": task, "arm": arm, "turn": turn, "model": model, "ref_source": source,
               "ref_sha256": hashlib.sha256(ref.encode()).hexdigest(), "prompt_sha256": sha, "seed": seed, "temperature": 0,
               "thinking": False, "num_ctx": NUM_CTX, "max_tokens": MAX_TOKENS, "finish": r.get("done_reason"),
               "prompt_tokens": r.get("prompt_eval_count"), "output_tokens": r.get("eval_count"), "gen_s": round(time.time() - t0, 1),
               "has_thinking": bool(r.get("message", {}).get("thinking")), "response": text, "code": extract_code(text)}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"[{i + 1}/{len(jobs)}] {row['id']}: {row['output_tokens']} tokens, {row['gen_s']} s, finish {row['finish']}, "
            f"{'code' if row['code'] else 'NO CODE'}")
    return len(jobs)


def items(model: str) -> list:
    """Oracle items for every attempt (both arms) that has no verdict yet. The code goes in as extracted, empty or not: the oracle's
    verdict on it (a reply with no code block fails as not having a workload) is the feedback the retry sees."""
    all_tasks, vs, out = tasks(), verdicts(model), []
    for arm in ARMS:
        for r in read_jsonl(run_dir(model) / f"{arm}.jsonl"):
            if r["id"] not in vs:
                ref = all_tasks[r["task"]][0]
                if hashlib.sha256(ref.encode()).hexdigest() != r["ref_sha256"]:
                    raise SystemExit(f"{r['id']}: the task's reference changed since it was generated")
                out.append({"id": r["id"], "task": r["task"], "ref_code": ref, "code": r["code"]})
    return out


def ingest(model: str, paths: list) -> int:
    known = {r["id"] for arm in ARMS for r in read_jsonl(run_dir(model) / f"{arm}.jsonl")}
    rows = [r for p in paths for r in read_jsonl(pathlib.Path(p)) if not r.get("meta")]
    unknown = [r["id"] for r in rows if r["id"] not in known]
    if unknown:
        raise SystemExit(f"{len(unknown)} result rows are not attempts of this run, e.g. {unknown[:3]}")
    with (run_dir(model) / "verdicts.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def summary(model: str) -> dict:
    """Per arm: verdicts at turn 0 and at each task's last judged turn, pass@1 and pass within 3 turns, and speed of the correct ones."""
    vs, all_tasks, out = verdicts(model), tasks(), {}
    for arm in ARMS:
        rows = read_jsonl(run_dir(model) / f"{arm}.jsonl")
        by = collections.defaultdict(list)
        for r in rows:
            if r["id"] in vs:
                by[r["task"]].append(vs[r["id"]] | {"turn": r["turn"]})
        first = {t: next(v for v in sorted(x, key=lambda v: v["turn"]) if v["turn"] == 0) for t, x in by.items() if any(v["turn"] == 0 for v in x)}
        last = {t: max(x, key=lambda v: v["turn"]) for t, x in by.items()}
        best = {t: next((v for v in x if v["verdict"] == "pass"), last[t]) for t, x in by.items()}
        correct = [v for v in best.values() if v["verdict"] in ("pass", "correct_slow") and v.get("speed_vs_xla") is not None]
        speeds = [v["speed_vs_xla"] for v in correct]
        out[arm] = {"tasks": len(all_tasks), "judged": len(by),
                    "pass_at_1": sum(v["verdict"] == "pass" for v in first.values()),
                    "pass_within_3": sum(v["verdict"] == "pass" for v in best.values()),
                    "correct_within_3": len(correct), "parity_within_3": sum(s >= 0.97 for s in speeds),
                    "median_speed_correct": statistics.median(speeds) if speeds else None,
                    "turn0_verdicts": dict(collections.Counter(v["verdict"] for v in first.values())),
                    "final_verdicts": dict(collections.Counter(v["verdict"] for v in best.values()))}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=MODEL)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    g = sub.add_parser("generate"); g.add_argument("--arm", choices=ARMS, required=True)
    g.add_argument("--tasks", nargs="*", default=()); g.add_argument("--limit", type=int, default=0)
    i = sub.add_parser("items"); i.add_argument("--out")
    n = sub.add_parser("ingest"); n.add_argument("results", nargs="+")
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "tasks":
        for t, (_, src) in tasks().items():
            print(f"{t:60} {src}")
        print(f"{len(tasks())} tasks")
    elif a.cmd == "generate":
        print(f"{generate(a.arm, a.model, a.tasks, a.limit)} attempts generated")
    elif a.cmd == "items":
        rows = items(a.model)
        text = "".join(json.dumps(r) + "\n" for r in rows)
        pathlib.Path(a.out).write_text(text) if a.out else sys.stdout.write(text)
        print(f"{len(rows)} items waiting for the oracle", file=sys.stderr)
    elif a.cmd == "ingest":
        print(f"{ingest(a.model, a.results)} verdicts ingested")
    else:
        print(json.dumps(summary(a.model), indent=1))


if __name__ == "__main__":
    main()
