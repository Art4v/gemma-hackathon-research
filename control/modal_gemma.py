"""Gemma on Modal GPUs, through the same Ollama and the same model file as the Mac, so every arm can be generated on one backend.

  modal run control/modal_gemma.py::pull      # once: the model into the "gemma-bench-ollama" volume, and its digest printed
  modal deploy control/modal_gemma.py         # the app bench.py --backend modal calls; containers scale to zero after 2 idle minutes

Each container runs one Ollama server with one request at a time (OLLAMA_NUM_PARALLEL=1), so an answer never shares a batch with another
and temperature 0 stays deterministic on a given GPU type. Parallelism comes from several containers (MAX_CONTAINERS), all the same GPU.
Flash attention stays off, as on the Mac. Every reply carries the backend it ran on: GPU, Ollama version, model digest.
"""
import json, os, subprocess, time, urllib.request

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
        r = _api("/api/chat", body)
        tags = {m["name"]: m["digest"] for m in _api("/api/tags")["models"]}
        digest = tags.get(body["model"]) or tags.get(f"{body['model']}:latest")  # the model that answered, not always MODEL (LoRA arms)
        return {**r, "backend": {**self.backend, "digest": digest, "model": body["model"]}}
