"""Constructor backend: Anthropic Message Batches (fallback for throughput). Needs `pip install anthropic` and ANTHROPIC_API_KEY.

Structured output is requested with output_config.format (JSON schema); adaptive thinking is left at the model's default. One batch
per generate() call; results are keyed by custom_id, never by position. Batches run asynchronously at half price; this backend
polls until the batch ends (poll interval 30 s), so a call can take up to the batch SLA. Verify the SDK surface against the
claude-api skill before relying on this in Phase 2: the API drifts and this file was written without running it.
"""
from __future__ import annotations

import json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from constructor import Request, Response  # noqa: E402


class AnthropicBatch:
    name = "anthropic_batch"

    def __init__(self, model: str = "claude-opus-5-5", poll_s: int = 30, effort: str = "high"):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pip install anthropic (dataset/.venv) to use the anthropic_batch backend") from e
        self.model, self.poll_s, self.effort = model, poll_s, effort

    def generate(self, requests: list) -> list:
        import anthropic
        client = anthropic.Anthropic()
        reqs = [{"custom_id": r.key, "params": {"model": self.model, "max_tokens": r.max_tokens, "system": r.system,
                                                  "messages": [{"role": "user", "content": r.user}],
                                                  "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": r.schema}}}}
                for r in requests]
        batch = client.messages.batches.create(requests=reqs)
        t0 = time.time()
        while batch.processing_status != "ended":
            time.sleep(self.poll_s)
            batch = client.messages.batches.retrieve(batch.id)
        wall = time.time() - t0
        by_key = {r.key: r for r in requests}
        out = []
        for res in client.messages.batches.results(batch.id):
            key, r = res.custom_id, res.result
            raw = json.dumps(res.model_dump(), default=str)
            if r.type != "succeeded":
                out.append(Response(key, False, None, self.model, self.name, {"wall_s": wall}, raw, error=f"batch result {r.type}"))
                continue
            msg = r.message
            text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
            usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens, "wall_s": wall, "cost_usd": None}
            try:
                parsed = json.loads(text)
                ok = isinstance(parsed, dict) and all(k in parsed for k in by_key[key].schema.get("required", []))
            except json.JSONDecodeError:
                parsed, ok = None, False
            out.append(Response(key, ok, parsed if ok else None, msg.model, self.name, usage, raw, error=None if ok else "unparseable output"))
        seen = {o.key for o in out}
        out += [Response(k, False, None, self.model, self.name, {}, "", error="missing from batch results") for k in by_key if k not in seen]
        return out
