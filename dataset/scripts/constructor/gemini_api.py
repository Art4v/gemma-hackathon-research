"""Constructor backend: the Gemini API through the google-genai SDK (`pip install google-genai`).

Auth comes from the environment, as the SDK reads it: GEMINI_API_KEY (use a paid key: the free tier may train on requests), or Vertex
AI with GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION. Each request is one generate_content call with
the system prompt as system_instruction and the schema as response_json_schema. Requests run in a thread pool; 429 and 5xx retry with
backoff. The raw response is saved in the same shape claude_cli uses ({"structured_output": ..., "usage": ...}), so --reuse-raw resumes
a Gemini run the same way.
"""
from __future__ import annotations

import concurrent.futures as cf, json, pathlib, random, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from constructor import Request, Response  # noqa: E402

DEFAULT_MODEL = "gemini-3.1-pro-preview"
RETRY_CODES = {408, 429, 500, 502, 503, 504}
# Gemini 3 counts thinking against max_output_tokens; Request.max_tokens (16k) was sized for the answer alone.
# ponytail: one fixed floor; raise it if responses come back with finish_reason MAX_TOKENS
MIN_OUTPUT_TOKENS = 65536


class GeminiAPI:
    name = "gemini"

    def __init__(self, model: str | None = None, workers: int = 3, retries: int = 4, effort: str | None = None, client=None):
        self.model, self.workers, self.retries = model or DEFAULT_MODEL, workers, retries
        self.thinking_level = effort or "high"  # generate.py's --effort maps onto Gemini's thinking_level (low, medium, high)
        if client is None:
            try:
                from google import genai
            except ImportError as e:
                raise RuntimeError("pip install google-genai (dataset/.venv) to use the gemini backend") from e
            client = genai.Client()
        self.client = client

    def _config(self, req: Request) -> dict:
        return {"system_instruction": req.system, "response_mime_type": "application/json", "response_json_schema": req.schema,
                "max_output_tokens": max(req.max_tokens, MIN_OUTPUT_TOKENS), "thinking_config": {"thinking_level": self.thinking_level}}

    def _one(self, req: Request) -> Response:
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                r = self.client.models.generate_content(model=self.model, contents=req.user, config=self._config(req))
            except Exception as e:  # the SDK raises errors.APIError with .code; anything else is not retried
                last, code = f"{type(e).__name__}: {e}", getattr(e, "code", None)
                if attempt < self.retries and code in RETRY_CODES:
                    time.sleep(min(120, 10 * 2 ** attempt) + random.random() * 5)
                    continue
                return Response(req.key, False, None, self.model, self.name, {"wall_s": round(time.time() - t0, 2)}, "", error=last[:500])
            return self._parse(req, r, time.time() - t0)
        return Response(req.key, False, None, self.model, self.name, {}, "", error=last[:500])

    def _parse(self, req: Request, r, wall: float) -> Response:
        um = getattr(r, "usage_metadata", None)
        out_tok = (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        usage = {"input_tokens": getattr(um, "prompt_token_count", 0) or 0, "output_tokens": out_tok, "wall_s": round(wall, 2), "cost_usd": None}
        model = getattr(r, "model_version", None) or self.model
        cands = getattr(r, "candidates", None) or []
        finish = str(getattr(cands[0], "finish_reason", "")) if cands else "no candidates"
        text = r.text or ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        ok = isinstance(parsed, dict) and all(k in parsed for k in req.schema.get("required", []))
        raw = json.dumps({"structured_output": parsed if ok else None, "usage": usage, "model": model, "finish_reason": finish, "text": text})
        if ok:
            return Response(req.key, True, parsed, model, self.name, usage, raw)
        return Response(req.key, False, None, model, self.name, usage, raw, error=f"unparseable output (finish_reason {finish})")

    def generate(self, requests: list) -> list:
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._one, requests))
