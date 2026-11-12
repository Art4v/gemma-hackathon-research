"""Greedy generation from a run's merged bf16 weights (sera-weights:/models/runs/<run>/merged), for checking that the Q4_0 Ollama model
built by to_ollama.py behaves like the LoRA it came from. Same prompts as the benchmark (control/bench.py conversation(), bare arm),
rendered with the checkpoint's own chat template, thinking off, greedy, at most 8192 new tokens.

  MODAL_PROFILE=splashgaming532 modal run training/bf16_generate.py --run pilot25_r64_e3 --prompts PROMPTS.json --out OUT.json
PROMPTS.json: {task: messages}. OUT.json: {task: {"text", "tokens", "prompt_tokens"}}.
"""
import json, pathlib

import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.15.0", "transformers==5.22.1", "accelerate==1.15.0", "safetensors==0.8.0", "jinja2"))
app = modal.App("lora-bf16-generate", image=image)
weights = modal.Volume.from_name("sera-weights")


@app.function(gpu="A100-80GB", volumes={"/models": weights}, timeout=3600)
def generate(run: str, prompts: dict, max_new: int = 8192) -> dict:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    path = f"/models/runs/{run}/merged"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMultimodalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    out = {}
    for task, messages in prompts.items():
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
        ids = {k: v.to("cuda") for k, v in ids.items()}
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        new = g[0, ids["input_ids"].shape[1]:]
        out[task] = {"text": tok.decode(new, skip_special_tokens=True), "tokens": int(new.shape[0]), "prompt_tokens": int(ids["input_ids"].shape[1])}
        print(task, out[task]["prompt_tokens"], out[task]["tokens"], flush=True)
    return out


@app.local_entrypoint()
def main(run: str, prompts: str, out: str):
    res = generate.remote(run, json.loads(pathlib.Path(prompts).read_text()))
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
