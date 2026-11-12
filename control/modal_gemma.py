"""Gemma on Modal GPUs, through the same Ollama and the same model file as the Mac, so every arm can be generated on one backend.

  modal run control/modal_gemma.py::pull      # once: the model into the "gemma-bench-ollama" volume, and its digest printed
  modal deploy control/modal_gemma.py         # the app bench.py --backend modal calls; containers scale to zero after 2 idle minutes

Each container runs one Ollama server with one request at a time (OLLAMA_NUM_PARALLEL=1), so an answer never shares a batch with another
and temperature 0 stays deterministic on a given GPU type. Parallelism comes from several containers (MAX_CONTAINERS), all the same GPU.
Flash attention stays off, as on the Mac. Every reply carries the backend it ran on: GPU, Ollama version, model digest.
"""
import json, os, subprocess, time, urllib.error, urllib.request

import modal

OLLAMA_VERSION = "0.40.14"  # the Mac's version (ollama --version, Nov 8)
MODEL = "gemma4:e4b-it-qat"
GPU = "A10G"  # one type for every container (L4 had no capacity on Nov 8)
MAX_CONTAINERS = 6
MODELS_DIR = "/ollama-models"

image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "ca-certificates", "zstd")
         .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh"))
volume = modal.Volume.from_name("gemma-bench-ollama", create_if_missing=True)
app = modal.App("gemma-bench", image=image)
ENV = {"OLLAMA_MODELS": MODELS_DIR, "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_FLASH_ATTENTION": "0", "OLLAMA_HOST": "127.0.0.1:11434",
       "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_MAX_LOADED_MODELS": "1"}


def _serve():
    proc = subprocess.Popen(["ollama", "serve"], env={**os.environ, **ENV}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
            return proc
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("ollama serve did not start")


def _api(path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:11434{path}", json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _chat_streamed(body: dict) -> dict:
    """/api/chat, streamed and assembled into the stream=False shape. The runner samples the same tokens either way; streaming only
    changes delivery. It matters when Ollama aborts a looping reply ("prediction aborted, token repeat limit reached", Nov 10, v4 on
    62p): unstreamed, that is a bare HTTP 500 and the reply is lost; streamed, the reply up to the abort is kept and recorded with
    done_reason "repeat_limit", like a reply cut at the token cap, so it is judged on what the model wrote (PROTOCOL: code goes to the
    oracle as is). Any other error is raised as text (urllib's HTTPError can't cross Modal's pickling)."""
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps({**body, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    content, thinking, tool_calls, final, model, chunks = [], [], [], None, None, 0
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for line in resp:
                c = json.loads(line)
                model = c.get("model", model)
                if "error" in c:
                    if "repeat limit" not in c["error"]:
                        raise RuntimeError(f"ollama: {c['error']}")
                    final = {"done": True, "done_reason": "repeat_limit", "error": c["error"], "eval_count": chunks}
                    break
                m = c.get("message") or {}
                content.append(m.get("content") or ""); thinking.append(m.get("thinking") or "")
                tool_calls += m.get("tool_calls") or []
                chunks += 1
                if c.get("done"):
                    final = {k: v for k, v in c.items() if k != "message"}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ollama HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}") from None
    if final is None:
        raise RuntimeError("ollama stream ended without a final chunk")
    msg = {"role": "assistant", "content": "".join(content)}
    if any(thinking):
        msg["thinking"] = "".join(thinking)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {**final, "model": model, "message": msg}


def _digest():
    return next((m["digest"] for m in _api("/api/tags")["models"] if m["name"] == MODEL), None)


@app.function(volumes={MODELS_DIR: volume}, timeout=1800)
def pull():
    """Download the model into the volume once (no GPU needed), so serving containers only read it."""
    _serve()
    subprocess.run(["ollama", "pull", MODEL], env={**os.environ, **ENV}, check=True)
    volume.commit()
    print(json.dumps({"model": MODEL, "digest": _digest(), "ollama": _api("/api/version")["version"]}))


@app.cls(gpu=GPU, volumes={MODELS_DIR: volume}, timeout=3600, scaledown_window=120, max_containers=MAX_CONTAINERS)
class Gemma:
    @modal.enter()
    def start(self):
        self.proc = _serve()
        self.backend = {"gpu": GPU, "ollama": _api("/api/version")["version"], "digest": _digest()}
        if self.backend["digest"] is None:
            raise RuntimeError(f"{MODEL} is not in the volume: run `modal run control/modal_gemma.py::pull` first")

    @modal.method()
    def chat(self, body: dict) -> dict:
        r = _chat_streamed(body)  # token-identical to stream=False (checked Nov 10: fresh A10G, A/B/C = stored row, 696 tokens)
        tags = {m["name"]: m["digest"] for m in _api("/api/tags")["models"]}
        digest = tags.get(body["model"]) or tags.get(f"{body['model']}:latest")  # the model that answered, not always MODEL (LoRA arms)
        return {**r, "backend": {**self.backend, "digest": digest, "model": body["model"]}}
