"""A trained LoRA as an Ollama model that differs from the benchmark's base (gemma4:e4b-it-qat) only in the weights the LoRA trained.

Ollama 0.40.14 no longer loads LoRA adapters ("LoRA adapters are no longer supported", server/create.go), so the adapter is folded in:
for each of the 258 tensors the LoRA wraps (attention q/k/v/o and MLP gate/up/down, all Q4_0 in the official GGUF), the new tensor is
Q4_0(W + scale * B @ A), with W the bf16 base checkpoint the LoRA was trained on and B, A the adapter's own fp32 matrices. The bytes are
written into a copy of the official GGUF at the same offsets, so every other tensor, all metadata and the vision/audio projector stay
byte-identical to what the base arm ran. gguf-py's Q4_0 reproduces the official bytes exactly on a dequantize/quantize round trip.

Two numbers are printed per run:
  control   share of Q4_0 blocks where Q4_0(W) equals the official file, i.e. whether the official model is this checkpoint quantized
  survival  share of blocks that change once the LoRA is added (0 would mean 4-bit rounding erased the LoRA)

  export MODAL_PROFILE=splashgaming532
  modal run training/to_ollama.py --run pilot25_r64_e3 --tag gemma4-e4b-lora:pilot25
Then bench.py --backend modal --model <tag> generates with it (control/modal_gemma.py serves any model in the gemma-bench-ollama volume).
"""
import json, os, pathlib, re, subprocess, time

import modal

BASE_TAG = "gemma4:e4b-it-qat"
BASE_DIGEST = "e8b6a059ba86947a44ace84d6e5679795bc41862c25c30513142588f0e9dba1d"  # the text model blob the base arm ran (ollama show)
HF_BASE = "/models/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/476025a01dbf99361c062bbeca3d6a76bb4c4566"
OLLAMA_VERSION = "0.40.14"
OM = "/ollama-models"
GGUF_TO_HF = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
              "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}

image = (modal.Image.debian_slim(python_version="3.12").apt_install("curl", "ca-certificates", "zstd")
         .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh")
         .uv_pip_install("gguf==0.19.0", "numpy"))
app = modal.App("lora-to-ollama", image=image)
weights = modal.Volume.from_name("sera-weights")
ollama_vol = modal.Volume.from_name("gemma-bench-ollama")


class SafeTensors:
    """Read tensors from (sharded) safetensors as float32 numpy without torch; bf16 is widened bit-exactly."""

    def __init__(self, folder):
        folder = pathlib.Path(folder)
        idx = folder / "model.safetensors.index.json"
        files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else \
            [p.name for p in folder.glob("*.safetensors")]
        self.where = {}
        for f in files:
            with open(folder / f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(n))
            for k, v in header.items():
                if k != "__metadata__":
                    self.where[k] = (folder / f, 8 + n, v)

    def get(self, key):
        import numpy as np
        path, base, v = self.where[key]
        a, b = v["data_offsets"]
        with open(path, "rb") as fh:
            fh.seek(base + a)
            raw = fh.read(b - a)
        if v["dtype"] == "BF16":
            return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
        assert v["dtype"] == "F32", v["dtype"]
        return np.frombuffer(raw, np.float32).reshape(v["shape"])


def ollama(*args, check=True):
    env = {**os.environ, "OLLAMA_MODELS": OM, "OLLAMA_HOST": "127.0.0.1:11434"}
    return subprocess.run(["ollama", *args], env=env, check=check, capture_output=True, text=True)


def serve():
    import urllib.request
    env = {**os.environ, "OLLAMA_MODELS": OM, "OLLAMA_HOST": "127.0.0.1:11434"}
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("ollama serve did not start")


def base_layers():
    """{mediaType: digest} of the base tag's manifest in the volume."""
    m = pathlib.Path(OM) / "manifests/registry.ollama.ai/library/gemma4/e4b-it-qat"
    man = json.loads(m.read_text())
    return {l["mediaType"]: l["digest"].split(":")[1] for l in man["layers"]}, man


@app.function(volumes={"/models": weights, OM: ollama_vol}, cpu=8, memory=32768, timeout=3600)
def build(run: str, tag: str):
    import numpy as np, gguf
    layers, _ = base_layers()
    model_digest = layers["application/vnd.ollama.image.model"]
    assert model_digest == BASE_DIGEST, f"the volume's {BASE_TAG} is {model_digest}, not the blob the base arm ran"
    projector = layers.get("application/vnd.ollama.image.projector")
    src = pathlib.Path(OM) / "blobs" / f"sha256-{model_digest}"
    adapter_dir = pathlib.Path(f"/models/runs/{run}/adapter")
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    base, ad = SafeTensors(HF_BASE), SafeTensors(adapter_dir)

    staging = pathlib.Path(OM) / "staging"
    staging.mkdir(exist_ok=True)
    out = staging / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', tag)}.gguf"
    t0 = time.time()
    subprocess.run(["cp", str(src), str(out)], check=True)
    reader = gguf.GGUFReader(str(src))
    stats = {"tensors": 0, "blocks": 0, "control_same_blocks": 0, "changed_blocks": 0}
    per_kind = {}
    with open(out, "r+b") as fh:
        for t in reader.tensors:
            m = re.fullmatch(r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight", t.name)
            if not m:
                continue
            assert t.tensor_type == gguf.GGMLQuantizationType.Q4_0, (t.name, t.tensor_type)
            hf = f"model.language_model.layers.{m.group(1)}.{GGUF_TO_HF[m.group(2)]}"
            W = base.get(hf + ".weight")
            A = ad.get(f"base_model.model.{hf}.lora_A.weight")
            B = ad.get(f"base_model.model.{hf}.lora_B.weight")
            official = np.asarray(t.data).view(np.uint8).reshape(W.shape[0], -1)
            assert official.shape[1] == W.shape[1] // 32 * 18, (t.name, official.shape, W.shape)
            q_base = gguf.quants.quantize(W, t.tensor_type).view(np.uint8).reshape(official.shape)
            q_new = gguf.quants.quantize(W + scale * (B @ A), t.tensor_type).view(np.uint8).reshape(official.shape)
            ob, bb, nb = (x.reshape(x.shape[0], -1, 18) for x in (official, q_base, q_new))
            same = int((ob == bb).all(-1).sum())
            changed = int((nb != ob).any(-1).sum())
            nblocks = ob.shape[0] * ob.shape[1]
            stats["tensors"] += 1; stats["blocks"] += nblocks; stats["control_same_blocks"] += same; stats["changed_blocks"] += changed
            k = per_kind.setdefault(m.group(2), [0, 0, 0]); k[0] += nblocks; k[1] += same; k[2] += changed
            fh.seek(t.data_offset)
            fh.write(q_new.tobytes())
    assert stats["tensors"] == len(cfg["target_modules"]) == 258, (stats["tensors"], len(cfg["target_modules"]))
    stats["control"] = round(stats["control_same_blocks"] / stats["blocks"], 6)
    stats["survival"] = round(stats["changed_blocks"] / stats["blocks"], 6)
    stats["per_kind"] = {k: {"control": round(v[1] / v[0], 6), "survival": round(v[2] / v[0], 6)} for k, v in per_kind.items()}
    stats["rewrite_s"] = round(time.time() - t0)
    print("STATS " + json.dumps(stats), flush=True)

    serve()
    mf = staging / "Modelfile"
    lines = [f"FROM {out}"] + ([f"FROM {pathlib.Path(OM) / 'blobs' / ('sha256-' + projector)}"] if projector else []) + \
            ["TEMPLATE {{ .Prompt }}", "RENDERER gemma4", "PARSER gemma4",
             "PARAMETER temperature 1", "PARAMETER top_k 64", "PARAMETER top_p 0.95"]
    mf.write_text("\n".join(lines) + "\n")
    r = ollama("create", tag, "-f", str(mf), check=False)
    print("ollama create:", r.returncode, (r.stdout + r.stderr)[-400:], flush=True)
    assert r.returncode == 0
    out.unlink(); mf.unlink()
    show = json.loads(subprocess.run(["curl", "-s", "127.0.0.1:11434/api/show", "-d", json.dumps({"model": tag})],
                                     capture_output=True, text=True).stdout)
    base_show = json.loads(subprocess.run(["curl", "-s", "127.0.0.1:11434/api/show", "-d", json.dumps({"model": BASE_TAG})],
                                          capture_output=True, text=True).stdout)
    for key in ("renderer", "parser", "capabilities"):
        print(f"{key}: lora {show.get(key)} | base {base_show.get(key)}", flush=True)
    assert show.get("renderer") == base_show.get("renderer") and show.get("parser") == base_show.get("parser")
    tags = {m["name"]: m["digest"] for m in json.loads(subprocess.run(["curl", "-s", "127.0.0.1:11434/api/tags"],
                                                                       capture_output=True, text=True).stdout)["models"]}
    ollama_vol.commit()
    print("DONE " + json.dumps({"tag": tag, "digest": tags.get(tag) or tags.get(f"{tag}:latest"), "base_digest": tags.get(BASE_TAG), **stats}))


@app.local_entrypoint()
def main(run: str, tag: str):
    # the workspace the benchmark generates in (BENCH_WORKSPACE; splashgaming532 for the base runs, codecraftacademy981 from Nov 9)
    ws = os.environ.get("BENCH_WORKSPACE", "splashgaming532")
    assert os.environ.get("MODAL_PROFILE") == ws, f"the benchmark's workspace is {ws}: set MODAL_PROFILE={ws}"
    build.remote(run, tag)
