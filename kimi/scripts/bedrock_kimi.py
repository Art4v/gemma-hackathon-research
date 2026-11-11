"""Constructor backend: Moonshot Kimi K3 on Amazon Bedrock (Converse API, boto3). Same protocol as gemini_api.py / claude_cli.py:
Request{key, system, user, schema{plan, code}} -> Response{ok, output, model, backend, usage, raw}. Kimi's reasoning text (Converse
`reasoningContent` blocks) is kept in output["reasoning"], so a dataset row can carry reasoning + plan + code.

Spend guard: every response's token usage is priced (PRICE_IN / PRICE_OUT per 1M tokens, set from the Bedrock pricing page; the
defaults are deliberately HIGH so the guard trips early) and added to a ledger file; no new request starts once the ledger plus the
worst case of the requests in flight would pass CAP_USD.
"""
from __future__ import annotations

import concurrent.futures as cf, json, os, pathlib, re, sys, threading, time

import boto3
from botocore.config import Config

sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/scripts")
from constructor import Request, Response  # noqa: E402  (the pipeline's own protocol)

JSON_RULE = ("\n\nOUTPUT FORMAT: reply with exactly one JSON object and nothing else, with two string fields: "
             '"plan" (3-6 lines: grid and block shapes, accumulator dtype, VMEM budget, what stays outside the kernel) and '
             '"code" (the complete Python module). No markdown fences around the JSON.')
FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)
PYFENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def _parse(text: str, required: list) -> dict | None:
    """The answer as {plan, code}: strict JSON, else a fenced JSON block, else the outermost {...}; last resort a python fence."""
    for cand in (text, *(m.group(1) for m in FENCE.finditer(text)), text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and all(k in d and isinstance(d[k], str) for k in required):
                return d
        except (json.JSONDecodeError, TypeError):
            pass
    m = PYFENCE.findall(text)
    if m and "code" in required:
        return {"plan": "", "code": m[-1], "_parsed_from": "python_fence"}
    # repair turns ask for "code only": Kimi then answers with the bare module (render_chat.unfence accepts the same)
    if "code" in required and "def workload" in text:
        return {"plan": "", "code": text.strip("\n") + "\n", "_parsed_from": "raw_module"}
    return None


class Ledger:
    def __init__(self, path: pathlib.Path, cap_usd: float):
        self.path, self.cap, self.lock = path, cap_usd, threading.Lock()
        self.spent = sum(json.loads(l)["cost_usd"] for l in path.open()) if path.exists() else 0.0

    def add(self, key: str, cost: float, usage: dict):
        with self.lock:
            self.spent += cost
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "cost_usd": round(cost, 6), **usage, "t": time.time()}) + "\n")


class BedrockKimi:
    name = "bedrock_kimi"

    def __init__(self, model: str = "global.moonshotai.kimi-k3.1", region: str = "us-east-1", workers: int = 16,
                 max_output_tokens: int = 32000, reasoning_fields: dict | None = None, cap_usd: float = 50.0,
                 price_in: float = 4.0, price_out: float = 20.0, price_cache_read: float = 0.4,
                 price_cache_write: float = 5.0, ledger: str = "ledger.jsonl", temperature: float | None = None):
        self.model, self.workers, self.max_out, self.reasoning_fields = model, workers, max_output_tokens, reasoning_fields
        self.price_in, self.price_out, self.temperature = price_in, price_out, temperature
        self.price_cr, self.price_cw = price_cache_read, price_cache_write
        self.ledger = Ledger(pathlib.Path(ledger), cap_usd)
        self.client = boto3.client("bedrock-runtime", region_name=region, config=Config(
            retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=1800, connect_timeout=30, max_pool_connections=workers + 4))
        self.worst = 30_000 / 1e6 * price_cache_write + max_output_tokens / 1e6 * price_out  # reserve per request in flight

    def _one(self, req: Request) -> Response:
        t = time.time()
        kw = dict(modelId=self.model, system=[{"text": req.system + JSON_RULE}],
                  messages=[{"role": "user", "content": [{"text": req.user}]}],
                  inferenceConfig={"maxTokens": self.max_out, **({"temperature": self.temperature} if self.temperature is not None else {})})
        if self.reasoning_fields:
            kw["additionalModelRequestFields"] = self.reasoning_fields
        try:
            r = self.client.converse(**kw)
        except Exception as e:  # botocore already retried throttling/5xx adaptively
            return Response(req.key, False, None, self.model, self.name, {"wall_s": round(time.time() - t, 1)}, "", error=f"{type(e).__name__}: {str(e)[:400]}")
        u = r.get("usage", {})
        usage = {"input_tokens": u.get("inputTokens"), "output_tokens": u.get("outputTokens"),
                 "cache_read_tokens": u.get("cacheReadInputTokens"), "cache_write_tokens": u.get("cacheWriteInputTokens"),
                 "wall_s": round(time.time() - t, 1),
                 "stop_reason": r.get("stopReason")}
        cost = ((usage["input_tokens"] or 0) * self.price_in + (usage["output_tokens"] or 0) * self.price_out
                + (usage["cache_read_tokens"] or 0) * self.price_cr + (usage["cache_write_tokens"] or 0) * self.price_cw) / 1e6
        usage["cost_usd"] = round(cost, 6)
        self.ledger.add(req.key, cost, usage)
        blocks = r.get("output", {}).get("message", {}).get("content", [])
        reasoning = "".join((b.get("reasoningContent", {}).get("reasoningText", {}) or {}).get("text", "") for b in blocks)
        text = "".join(b.get("text", "") for b in blocks if "text" in b)
        raw = json.dumps(r, default=str)
        out = _parse(text, req.schema.get("required", ["plan", "code"]))
        if out is None:
            return Response(req.key, False, {"reasoning": reasoning, "text": text}, self.model, self.name, usage, raw,
                            error=f"unparseable answer (stop {usage['stop_reason']}, {len(text)} chars)")
        out["reasoning"] = reasoning
        return Response(req.key, True, out, self.model, self.name, usage, raw)

    def generate(self, requests: list, on_result=None) -> list:
        """Each result is handed to on_result (and printed) the moment it completes, so nothing is held only in memory. At most
        2 x workers requests are outstanding, and none is admitted unless ledger + outstanding worst case stays within the cap."""
        out, lock = [], threading.Lock()

        def run(r):
            try:
                resp = self._one(r)
            except Exception as e:  # never lose the slot
                resp = Response(r.key, False, None, self.model, self.name, {}, "", error=f"{type(e).__name__}: {str(e)[:300]}")
            with lock:
                out.append(resp)
                if on_result:
                    on_result(resp)
                print(f"  [{len(out)}/{len(requests)}] {resp.key} {'ok' if resp.ok else 'FAIL ' + str(resp.error)[:100]} "
                      f"in {resp.usage.get('input_tokens')} out {resp.usage.get('output_tokens')} ${self.ledger.spent:.2f}", flush=True)
            return resp

        with cf.ThreadPoolExecutor(self.workers) as ex:
            futs = []
            for req in requests:
                while True:
                    outstanding = sum(1 for f in futs if not f.done())
                    if outstanding < 2 * self.workers and self.ledger.spent + (outstanding + 1) * self.worst <= self.ledger.cap:
                        break
                    if outstanding == 0:
                        print(f"CAP REACHED: ${self.ledger.spent:.2f} of ${self.ledger.cap:.2f}; {len(requests) - len(futs)} requests not sent", flush=True)
                        return out
                    time.sleep(1)
                futs.append(ex.submit(run, req))
            cf.wait(futs)
        return out
