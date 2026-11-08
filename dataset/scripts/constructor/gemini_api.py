"""Constructor backend: Gemini API generateContent (REST, no SDK). Needs GEMINI_API_KEY (env, or the repo-root .env).

Each request is one call: system prompt as systemInstruction, the user message as one user turn, structured output via
responseMimeType=application/json + responseJsonSchema (the Request's JSON schema, unchanged). Requests run in a thread pool;
429 / 5xx / timeouts retry with exponential backoff. Thinking tokens count toward maxOutputTokens on Gemini, so the output budget
is the model's ceiling, not Request.max_tokens. Use a billed Cloud project key: unpaid-tier prompts and outputs may be used by Google.
"""
from __future__ import annotations

import concurrent.futures as cf, json, os, pathlib, random, sys, time, urllib.error, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from constructor import Request, Response  # noqa: E402

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_HTTP = {429, 500, 502, 503, 504}


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    for env in (pathlib.Path(__file__).resolve().parents[3] / ".env", pathlib.Path.home() / "Projects/gemma-hackathon-research/.env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GEMINI_API_KEY is not set (env or .env)")


class GeminiAPI:
    name = "gemini_api"

    def __init__(self, model: str | None = None, workers: int = 3, timeout_s: int = 900, retries: int = 4, effort: str | None = None,
                 max_output_tokens: int = 32768):
        self.model, self.workers, self.timeout_s, self.retries = model or "gemini-4-argon", workers, timeout_s, retries
        self.effort, self.max_out, self.key = effort, max_output_tokens, _key()

    def _body(self, req: Request) -> dict:
        gc = {"responseMimeType": "application/json", "responseJsonSchema": req.schema, "maxOutputTokens": self.max_out}
        if self.effort:  # a number is a thinking-token cap (thinkingBudget); a word is a thinkingLevel. argon accepts one, not both.
            # Nov 8 test: at thinkingLevel high, 2 of 8 kernels spent all 32K output tokens thinking and returned nothing (MAX_TOKENS).
            gc["thinkingConfig"] = {"thinkingBudget": int(self.effort)} if str(self.effort).isdigit() else {"thinkingLevel": self.effort}
        return {"systemInstruction": {"parts": [{"text": req.system}]}, "contents": [{"role": "user", "parts": [{"text": req.user}]}],
                "generationConfig": gc}

    def _one(self, req: Request) -> Response:
        last, raw = "", ""
        for attempt in range(self.retries + 1):
            t = time.time()
            try:
                r = urllib.request.Request(URL.format(model=self.model), data=json.dumps(self._body(req)).encode(),
                                           headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
                raw = urllib.request.urlopen(r, timeout=self.timeout_s).read().decode()
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}"
                if e.code in RETRY_HTTP and attempt < self.retries:
                    time.sleep(min(300, 20 * 2 ** attempt) + random.random() * 10)
                    continue
                break
            except Exception as e:  # timeouts, resets
                last = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    time.sleep(min(300, 20 * 2 ** attempt) + random.random() * 10)
                    continue
                break
            wall = round(time.time() - t, 1)
            d = json.loads(raw)
            um = d.get("usageMetadata") or {}
            u = {"input_tokens": um.get("promptTokenCount"), "output_tokens": um.get("candidatesTokenCount"),
                 "thinking_tokens": um.get("thoughtsTokenCount"), "wall_s": wall}
            model = d.get("modelVersion") or self.model
            cand = (d.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
            try:
                out = json.loads(text)
                if all(k in out for k in req.schema.get("required", [])):
                    return Response(req.key, True, out, model, self.name, u, raw)
                last = f"missing required keys (finishReason {cand.get('finishReason')})"
            except json.JSONDecodeError:
                last = f"unparseable output (finishReason {cand.get('finishReason')}, {len(text)} chars)"
            return Response(req.key, False, None, model, self.name, u, raw, error=last)
        return Response(req.key, False, None, self.model, self.name, {}, raw, error=last[:500])

    def generate(self, requests: list) -> list:
        out = []
        with cf.ThreadPoolExecutor(self.workers) as ex:
            futs = {ex.submit(self._one, r): r.key for r in requests}
            for f in cf.as_completed(futs):
                resp = f.result()
                out.append(resp)
                print(f"  [{len(out)}/{len(requests)}] {resp.key} {'ok' if resp.ok else 'FAIL ' + str(resp.error)[:120]} {resp.usage}", flush=True)
        return out
