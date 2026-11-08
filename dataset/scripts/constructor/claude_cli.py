"""Constructor backend: Claude Code CLI in headless mode (`claude -p`) under the user's subscription.

Each request is one process: the user message on stdin, the system prompt via --system-prompt, structured output via --json-schema,
no tools, no session persistence. Requests run in a thread pool; 429s and transient errors retry with backoff. The probe that fixed
these flags (Nov 8): `--bare` loses the login, so it is not used; the JSON result carries `structured_output` and `modelUsage`.
"""
from __future__ import annotations

import concurrent.futures as cf, json, os, pathlib, random, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from constructor import Request, Response  # noqa: E402

RETRY_MARKERS = ("rate limit", "429", "overloaded", "529", "timeout", "ECONNRESET", "Not logged in")


class ClaudeCLI:
    name = "claude_cli"

    def __init__(self, model: str | None = None, workers: int = 3, timeout_s: int = 900, retries: int = 3, effort: str | None = None,
                 claude_bin: str = "claude"):
        self.model, self.workers, self.timeout_s, self.retries, self.effort, self.bin = model or "default", workers, timeout_s, retries, effort, claude_bin

    def _argv(self, req: Request) -> list:
        argv = [self.bin, "-p", "--tools", "", "--no-session-persistence", "--output-format", "json", "--json-schema", json.dumps(req.schema),
                "--system-prompt", req.system]
        if self.model != "default":
            argv += ["--model", self.model]
        if self.effort:
            argv += ["--effort", self.effort]
        return argv

    def _one(self, req: Request) -> Response:
        last = ""
        for attempt in range(self.retries + 1):
            t0, p = time.time(), None
            try:
                p = subprocess.run(self._argv(req), input=req.user, capture_output=True, text=True, timeout=self.timeout_s,
                                   cwd=os.environ.get("CLAUDE_CLI_CWD", str(pathlib.Path(__file__).resolve().parents[2])))
                raw = p.stdout
                d = json.loads(raw) if raw.strip().startswith("{") else {}
            except subprocess.TimeoutExpired:
                raw, d, p = f'{{"is_error": true, "result": "timeout after {self.timeout_s}s"}}', {"is_error": True, "result": "timeout"}, None
            except json.JSONDecodeError as e:
                raw, d = (p.stdout if p else ""), {"is_error": True, "result": f"bad json: {e}: {(p.stderr if p else '')[-200:]}"}
            wall = time.time() - t0
            model = next(iter((d.get("modelUsage") or {}).keys()), self.model)
            usage = d.get("usage") or {}
            u = {"input_tokens": usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0),
                 "output_tokens": usage.get("output_tokens", 0), "wall_s": round(wall, 2), "cost_usd": d.get("total_cost_usd")}
            out = d.get("structured_output")
            if not d.get("is_error") and isinstance(out, dict):
                return Response(req.key, True, out, model, self.name, u, raw)
            last = str(d.get("result") or (p.stderr[-300:] if p else ""))
            if attempt < self.retries and any(m.lower() in last.lower() for m in RETRY_MARKERS):
                time.sleep(min(120, 10 * 2 ** attempt) + random.random() * 5)
                continue
            return Response(req.key, False, None, model, self.name, u, raw, error=last[:500])
        return Response(req.key, False, None, self.model, self.name, {}, "", error=last[:500])

    def generate(self, requests: list) -> list:
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._one, requests))
