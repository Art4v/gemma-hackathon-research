"""The Gemini Flash reference arm for the control benchmark: the same tasks, prompts (bench.conversation, docs arm), turns, retry rule
and oracle as the Gemma arms; only the model call differs. Gemini runs at its own defaults (thinking on, default temperature), its
strongest single-sample setting, with a 64k output budget because its thinking counts against it. Rows use bench.py's layout in
control/runs/<served model>/docs.jsonl, so `bench.py --model <served model> items|ingest|summary` work unchanged.

  GEMINI_API_KEY=... python control/bench_gemini.py generate [--model gemini-3.5-flash] [--tasks T ...] [--workers 8]

Note: the API serves "gemini-3.5-flash" as gemini-3.6-flash (Nov 8); rows record the served version.
"""
import argparse, concurrent.futures as cf, hashlib, json, os, pathlib, random, sys, time, urllib.error, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_OUT, ARM, RETRY = 65536, "docs", {408, 429, 500, 502, 503, 504}


def gemini_chat(messages: list, model: str, key: str, retries: int = 5) -> dict:
    """bench-style messages -> one generateContent call; returns text, finish, token counts and the served model."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": {"maxOutputTokens": MAX_OUT}}
    for attempt in range(retries + 1):
        req = urllib.request.Request(URL.format(model=model), json.dumps(body).encode(), {"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=900))
            break
        except urllib.error.HTTPError as e:
            if e.code in RETRY and attempt < retries:
                time.sleep(min(120, 10 * 2 ** attempt) + random.random() * 5); continue
            raise SystemExit(f"Gemini API error {e.code}: {e.read()[:300]!r}")
    c = (d.get("candidates") or [{}])[0]
    um = d.get("usageMetadata", {})
    text = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
    return {"text": text, "finish": c.get("finishReason"), "prompt_tokens": um.get("promptTokenCount"),
            "output_tokens": (um.get("candidatesTokenCount") or 0) + (um.get("thoughtsTokenCount") or 0),
            "thinking_tokens": um.get("thoughtsTokenCount"), "served": d.get("modelVersion") or model}


def generate(model: str, run_tag: str, only=(), workers: int = 8) -> int:
    key = os.environ.get("GEMINI_API_KEY", "").strip() or sys.exit("GEMINI_API_KEY is not set")
    all_tasks = bench.tasks()
    if only:
        all_tasks = {t: v for t, v in all_tasks.items() if t in only}
    out = bench.run_dir(run_tag) / f"{ARM}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = bench.pending(ARM, bench.read_jsonl(out), bench.verdicts(run_tag), all_tasks)
    docs, sha = bench.docs_for(ARM), bench.prompt_sha256(ARM)

    def one(job):
        task, turn, history = job
        ref, source = all_tasks[task]
        t0 = time.time()
        r = gemini_chat(bench.conversation(ref, history, docs), model, key)
        return {"id": bench.item_id(ARM, task, turn), "task": task, "arm": ARM, "turn": turn, "model": run_tag, "requested_model": model,
                "served_model": r["served"], "ref_source": source, "ref_sha256": hashlib.sha256(ref.encode()).hexdigest(), "prompt_sha256": sha,
                "seed": None, "temperature": "default", "thinking": "default", "num_ctx": None, "max_tokens": MAX_OUT, "finish": r["finish"],
                "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"], "thinking_tokens": r["thinking_tokens"],
                "gen_s": round(time.time() - t0, 1), "has_thinking": bool(r["thinking_tokens"]), "response": r["text"], "code": bench.extract_code(r["text"])}

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(one, jobs)):
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i + 1}/{len(jobs)}] {row['id']}: {row['output_tokens']} tokens ({row['thinking_tokens']} thinking), {row['gen_s']} s, "
                  f"finish {row['finish']}, {'code' if row['code'] else 'NO CODE'}", flush=True)
    return len(jobs)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", default="gemini-3.5-flash"); g.add_argument("--run-tag", default="gemini-3.6-flash")
    g.add_argument("--tasks", nargs="*", default=()); g.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    print(f"{generate(a.model, a.run_tag, a.tasks, a.workers)} attempts generated")


if __name__ == "__main__":
    main()
