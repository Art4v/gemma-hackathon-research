"""Register the v4 LoRA model in the gemma-bench Ollama volume, exactly as the Mac serves it (same GGUF bytes, same projector, same
Modelfile lines), and verify: GGUF sha256, projector present, and the resulting model digest vs the Mac's (2a99a2f97822).
  modal run control/modal_v4_create.py::create   (then ::verify; Ollama 0.40 writes each layer's source path into the manifest, so the
  Modal digest 0f764d3f6798 differs from the Mac's 2a99a2f97822 while every layer and the config blob are identical)"""
import hashlib, json, os, subprocess, time, urllib.request
import modal

OLLAMA_VERSION = "0.40.14"           # same as control/modal_gemma.py
MODELS_DIR = "/ollama-models"
BASE = "gemma4:e4b-it-qat"
NAME = "gemma4-e4b-lora:v4final-e2"
GGUF = f"{MODELS_DIR}/lora/v4final_e2_q8.gguf"
GGUF_SHA = "342ca4285212d52651de1e08dc3216e9118b78f87612817c450551e278d96411"
PROJ = f"{MODELS_DIR}/blobs/sha256-c6398448d84a4836fdedf58f9775979e69ae0cc4dfdf4d697b5597693a555b12"
MAC_DIGEST = "2a99a2f97822"
MODELFILE = f"""FROM {GGUF}
FROM {PROJ}
TEMPLATE {{{{ .Prompt }}}}
RENDERER gemma4
PARSER gemma4
PARAMETER temperature 1
PARAMETER top_k 64
PARAMETER top_p 0.95
"""

image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "ca-certificates", "zstd")
         .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh"))
volume = modal.Volume.from_name("gemma-bench-ollama")
app = modal.App("gemma-bench-setup", image=image)
ENV = {"OLLAMA_MODELS": MODELS_DIR, "OLLAMA_HOST": "127.0.0.1:11434"}


def _api(path):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:11434{path}", timeout=60))


MAC_LAYERS = {"model": f"sha256:{GGUF_SHA}", "projector": "sha256:" + PROJ.rsplit("sha256-", 1)[1],
              "template": "sha256:b507b9c2f6ca", "params": "sha256:56380ca2ab89"}  # the Mac manifest's layers (prefixes)


def _verify():
    tags = {m["name"]: m["digest"] for m in _api("/api/tags")["models"]}
    repo, tag = NAME.split(":")
    man = json.load(open(f"{MODELS_DIR}/manifests-v2/ollama.com/library/{repo}/{tag}"))  # Ollama 0.40 layout (as on the Mac)
    layers = {l["mediaType"].rsplit(".", 1)[-1]: l["digest"] for l in man["layers"]}
    for k, want in MAC_LAYERS.items():
        assert layers.get(k, "").startswith(want), f"{k} layer is {layers.get(k)}, the Mac has {want}"
    assert len(layers) == len(MAC_LAYERS), f"extra layers: {sorted(layers)}"
    return {"model_layer_is_v4_gguf": True, "all_layers_match_mac": True, "model": NAME, "digest": tags.get(NAME),
            "base_digest": tags.get(BASE), "matches_mac_digest": (tags.get(NAME) or "").startswith(MAC_DIGEST),
            "config": man["config"]["digest"][:19], "ollama": _api("/api/version")["version"]}


def _start():
    subprocess.Popen(["ollama", "serve"], env={**os.environ, **ENV}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            _api("/api/version"); return
        except OSError:
            time.sleep(0.5)


@app.function(volumes={MODELS_DIR: volume}, timeout=600)
def verify():
    _start()
    out = _verify()
    repo, tag = NAME.split(":")
    raw = open(f"{MODELS_DIR}/manifests-v2/ollama.com/library/{repo}/{tag}", "rb").read()
    print("MANIFEST", hashlib.sha256(raw).hexdigest()[:12], raw.decode())
    print(json.dumps(out))
    return out


@app.function(volumes={MODELS_DIR: volume}, timeout=1800, cpu=4, memory=16384)
def create():
    h = hashlib.sha256()
    with open(GGUF, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    assert h.hexdigest() == GGUF_SHA, f"GGUF sha mismatch: {h.hexdigest()[:12]}"
    assert os.path.exists(PROJ), "projector blob missing (pull the base model first)"
    subprocess.Popen(["ollama", "serve"], env={**os.environ, **ENV}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            _api("/api/version"); break
        except OSError:
            time.sleep(0.5)
    open("/tmp/Modelfile", "w").write(MODELFILE)
    subprocess.run(["ollama", "create", NAME, "-f", "/tmp/Modelfile"], env={**os.environ, **ENV}, check=True)
    volume.commit()
    out = _verify()
    print(json.dumps(out))
    return out
