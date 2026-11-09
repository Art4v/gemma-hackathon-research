"""Constructor backend: the Antigravity CLI (`agy -p`) under the user's Google AI subscription login (Nov 8: replaces Gemini CLI).

`agy` has no system-prompt or JSON-schema flag, so each request is one prompt: the system text, the user message, then an instruction
to reply with one JSON object matching the schema. Each call runs in a fresh empty temp directory (no GEMINI.md/AGENTS.md rules, no
repo files for the agent to read or edit) with `--output-format json`; the envelope's `response` is parsed as JSON (fences stripped).
Low concurrency by default: a personal plan's quota is shared with interactive use. Quota/rate errors retry with backoff.
"""
from __future__ import annotations

import concurrent.futures as cf, json, os, pathlib, random, re, subprocess, sys, tempfile, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from constructor import Request, Response  # noqa: E402

AGY = os.environ.get("AGY_BIN", str(pathlib.Path.home() / ".local" / "bin" / "agy"))
RETRY_MARKERS = ("quota", "rate limit", "429", "resource_exhausted", "overloaded", "503", "timeout", "unavailable")


def _parse(text: str):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    elif not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        t = t[i:j + 1] if i >= 0 and j > i else t
    return json.loads(t)


class AgyCLI:
    name = "agy_cli"

    def __init__(self, model: str | None = None, workers: int = 4, timeout_s: int = 900, retries: int = 4, effort: str | None = None):
        self.model, self.workers, self.timeout_s, self.retries = model or "default", workers, timeout_s, retries

    def _prompt(self, req: Request) -> str:
        return (req.system + "\n\n---\n\n" + req.user + "\n\n---\n\nReply with exactly one JSON object and nothing else (no prose, no "
                "code fences around it), matching this JSON schema:\n" + json.dumps(req.schema) +
                "\nDo not use any tools and do not create or edit files: answer directly.")

    def _one(self, req: Request) -> Response:
        last, raw = "", ""
        argv = [AGY, "-p", self._prompt(req), "--output-format", "json", "--disable-slash-commands"]
        if self.model != "default":
            argv += ["--model", self.model]
        for attempt in range(self.retries + 1):
            t = time.time()
            with tempfile.TemporaryDirectory(prefix="agy-") as d:
                try:
                    p = subprocess.run(argv, cwd=d, capture_output=True, text=True, timeout=self.timeout_s, stdin=subprocess.DEVNULL)
                except subprocess.TimeoutExpired:
                    last = f"timeout {self.timeout_s}s"
                    continue
            raw, wall = p.stdout, round(time.time() - t, 1)
            try:
                env = json.loads(p.stdout)
            except json.JSONDecodeError:
                env = None
            err = (p.stderr or "")[-400:] if env is None else json.dumps(env.get("error") or "")[:400]
            if env is None or p.returncode or (env.get("status") not in (None, "success", "ok", "completed")):
                last = f"rc {p.returncode}: {err or (p.stdout or '')[-300:]}"
                if any(m in last.lower() for m in RETRY_MARKERS) and attempt < self.retries:
                    time.sleep(min(300, 30 * 2 ** attempt) + random.random() * 10)
                    continue
                break
            u = env.get("usage") or {}
            usage = {"input_tokens": u.get("input_tokens") or u.get("promptTokenCount"), "output_tokens": u.get("output_tokens") or u.get("candidatesTokenCount"),
                     "thinking_tokens": u.get("thinking_tokens") or u.get("thoughtsTokenCount"), "wall_s": wall}
            model = env.get("model") or self.model
            try:
                out = _parse(env.get("response") or "")
                if all(k in out for k in req.schema.get("required", [])):
                    return Response(req.key, True, out, model, self.name, usage, raw)
                last = "missing required keys"
            except (json.JSONDecodeError, TypeError):
                last = "unparseable response"
            return Response(req.key, False, None, model, self.name, usage, raw, error=last)
        return Response(req.key, False, None, self.model, self.name, {}, raw, error=last[:500])

    def generate(self, requests: list) -> list:
        out = []
        with cf.ThreadPoolExecutor(self.workers) as ex:
            for f in cf.as_completed([ex.submit(self._one, r) for r in requests]):
                resp = f.result()
                out.append(resp)
                print(f"  [{len(out)}/{len(requests)}] {resp.key} {'ok' if resp.ok else 'FAIL ' + str(resp.error)[:120]} {resp.usage}", flush=True)
        return out
