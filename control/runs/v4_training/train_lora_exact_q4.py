"""LoRA fine-tune of Gemma 4 E4B on Modal for the JAXBench LoRA arm.

Copied from planningGemma cloud-gemma/sera-e4b/train_r64.py at f1e02e4 (origin/main, Nov 7). Changes, all small: --val takes a validation
file (Aarav's val split, rows from training/build_rows.py) in place of holding out every 20th task, and the examples below use this repo's
paths and the splashgaming532 workspace (the one the control benchmark ran in). Rows from build_rows.py carry thinking: false, as the
benchmark samples with thinking off. Everything else, including every preflight check, is planningGemma's.

  export MODAL_PROFILE=splashgaming532
  modal run training/train_lora.py --workspace splashgaming532 --step preflight --data training/data/pilot/train.jsonl
  modal deploy training/train_lora.py
  modal run training/train_lora.py --workspace splashgaming532 --step launch --data training/data/pilot/train.jsonl --val training/data/pilot/val.jsonl --name pilot_r64 --epochs 3

planningGemma's original notes:

LoRA fine-tune of Gemma 4 E4B or E2B on Modal, at the config context.md asks for: r = 64, alpha = 128, attention and MLP.

Adapted from practice/train_app.py, which is left as it was. A plain PyTorch loop, so every step is visible and the preflight can
check the traps that bit earlier runs: reasoning silently dropped by the chat template, a learning rate of 0, and LoRA landing in
the vision tower. New here: a parameter-count check against the config, teacher and reasoning filters, and E2B.

  MODAL_PROFILE=<ws> modal run train_r64.py --workspace <ws> --step preflight --base e4b --config r64 --data results/train.jsonl
  MODAL_PROFILE=<ws> modal deploy train_r64.py                      # once per workspace
  MODAL_PROFILE=<ws> modal run train_r64.py --workspace <ws> --step launch --base e4b --config r64 --data results/train.jsonl --name e4b_r64_ds --epochs 3
  MODAL_PROFILE=<ws> modal run train_r64.py --workspace <ws> --step status --call-id fc-...
--workspace is opt-in: it refuses an unset or `deepseek` MODAL_PROFILE (that workspace bills a card) and a profile other than the one named.
It runs inside main, so `modal run` has already created an empty ephemeral app in that profile's workspace when it refuses; no step runs. Set MODAL_PROFILE first.
`launch` spawns `train` on the deployed app and exits, so a laptop losing network cannot cancel the run (a plain `modal run --step train`
can be cancelled that way, detached or not; it cost two runs on Nov 5). Weights go to the sera-weights volume: /models/runs/<name>/{merged,adapter,metrics.json}.
"""
import json, os, pathlib, re, shutil, time

import modal

MODELS = "/models"
RUNS = f"{MODELS}/runs"
TURN_END_ID = 106  # <turn|>
BASES = {"e4b": ("google/gemma-4-E4B-it-qat-q4_0-unquantized", "476025a01dbf99361c062bbeca3d6a76bb4c4566"),
         "e2b": ("google/gemma-4-E2B-it-qat-q4_0-unquantized", "6befbaca7398925921802abd1f277b495b78b738")}
CONFIGS = {"r64": {"rank": 64, "alpha": 128, "targets": "attn_mlp"},      # context.md
           "r16": {"rank": 16, "alpha": 32, "targets": "attn_mlp"},       # rank sweep
           "r128": {"rank": 128, "alpha": 256, "targets": "attn_mlp"},
           "r64attn": {"rank": 64, "alpha": 128, "targets": "attn"}}
TARGET_RE = {"attn": re.compile(r".*language_model.*\.(q|k|v|o)_proj$"),
             "attn_mlp": re.compile(r".*language_model.*\.(q|k|v|o|gate|up|down)_proj$")}

volume = modal.Volume.from_name("sera-weights", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.15.0", "transformers==5.22.1", "peft==0.21.2", "accelerate==1.15.0", "safetensors==0.8.0", "hf_xet", "jinja2", "gguf", "numpy")
         .env({"HF_HOME": MODELS}))
app = modal.App("sera-train", image=image)


def is_target(module_name, targets):
    """Text layers only: PEFT refuses the vision tower's clipped linears, and the audio tower is not what we're teaching."""
    return bool(TARGET_RE[targets].match(module_name))


def lora_params(text_cfg, rank, targets):
    """Trainable parameters for rank r on the chosen projections, from the config alone. The preflight checks the real model against this."""
    h, n_layers = text_cfg["hidden_size"], text_cfg["num_hidden_layers"]
    shared_from = n_layers - text_cfg.get("num_kv_shared_layers", 0)
    total = 0
    for i, kind in enumerate(text_cfg["layer_types"]):
        # E2B's double-wide MLP applies to the kv-sharing layers only (measured: 96,632,832 at r = 64, see tests)
        inter = text_cfg["intermediate_size"] * (2 if text_cfg.get("use_double_wide_mlp") and i >= shared_from else 1)
        hd = text_cfg["global_head_dim"] if kind == "full_attention" else text_cfg["head_dim"]
        q_out, kv_out = text_cfg["num_attention_heads"] * hd, text_cfg["num_key_value_heads"] * hd
        total += rank * (h + q_out) * 2  # q_proj and o_proj
        if i < shared_from:
            total += rank * (h + kv_out) * 2  # k_proj and v_proj
        if targets == "attn_mlp":
            total += rank * (h + inter) * 3  # gate, up, down
    return total


def select_examples(rows, teachers=None, reasoning=None, near_test=True):
    """reasoning=True keeps only rows with reasoning; False blanks reasoning (code-only training); None leaves rows as they are.
    near_test=False drops the attention-like rows that sit closest to the test set's attention tasks."""
    out = []
    for r in rows:
        if teachers and r.get("teacher") not in teachers:
            continue
        if not near_test and r.get("near_test"):
            continue
        has = bool(r["messages"][-1].get("reasoning", "").strip())
        if reasoning is True and not has:
            continue
        r = json.loads(json.dumps(r))
        if reasoning is False:
            r["messages"][-1]["reasoning"] = ""
        out.append(r)
    return out


def encode(tok, messages, thinking=True):
    """Prompt and answer token ids, kept apart: the model only gets graded on the answer.
    thinking must match how the row will be evaluated: Gemma 4 adds <|think|> to the system turn only when it is on."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    full = tok.apply_chat_template(messages, tokenize=False, enable_thinking=thinking).rstrip("\n")
    assert full.startswith(prompt), "the training prompt must match what the model sees at inference"
    p = tok(prompt, add_special_tokens=False).input_ids
    a = tok(full[len(prompt):], add_special_tokens=False).input_ids
    assert a[-1] == TURN_END_ID, "the answer must end by closing the turn, or the model never learns to stop"
    if messages[-1].get("reasoning", "").strip():
        assert "<|channel>thought" in tok.decode(a[:12]), "reasoning was dropped by the chat template"
    return p, a


def answer_loss(model, p, a, device, chunk=2048):
    """Cross-entropy on the answer tokens only, with vocabulary scores built a chunk at a time under checkpointing."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    cap = core.config.text_config.final_logit_softcapping

    def chunk_loss(hidden, target):
        logits = core.lm_head(hidden).float()
        logits = torch.tanh(logits / cap) * cap
        return F.cross_entropy(logits, target, reduction="sum")

    ids = torch.tensor([p + a], device=device)
    hidden = core.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, -(len(a) + 1):-1]
    target = torch.tensor(a, device=device)
    total = 0.0
    for i in range(0, len(a), chunk):
        total = total + checkpoint(chunk_loss, hidden[i:i + chunk], target[i:i + chunk], use_reentrant=False)
    return total / len(a)



# ---- quantization-aware base (inlined from exact_q4.py) ----
EXACT_Q4 = True   # train on the exact weights Ollama serves (official gemma4:e4b-it-qat GGUF), not on the bf16 checkpoint
GGUF_URL = "https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:e8b6a059ba86947a44ace84d6e5679795bc41862c25c30513142588f0e9dba1d"
import hashlib
import numpy as np

GGUF_SHA256 = "e8b6a059ba86947a44ace84d6e5679795bc41862c25c30513142588f0e9dba1d"
MAP = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
       "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
       "inp_gate": "per_layer_input_gate", "proj": "per_layer_projection"}
Q4 = re.compile(r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down|inp_gate|proj)\.weight")
EXTRA = {"token_embd.weight": "model.language_model.embed_tokens.weight",
         "per_layer_token_embd.weight": "model.language_model.embed_tokens_per_layer.weight",
         "per_layer_model_proj.weight": "model.language_model.per_layer_model_projection.weight"}
EXPECTED = {"Q4_0": 342, "Q6_K": 2, "F16": 1}


def hf_name(gguf_name):
    m = Q4.fullmatch(gguf_name)
    if m:
        return f"model.language_model.layers.{int(m.group(1))}.{MAP[m.group(2)]}.weight"
    return EXTRA.get(gguf_name)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def apply_exact_base(model, blob, check_sha=True, rows_per_chunk=4096, strict=True):
    """In place, before the LoRA is added. Returns {gguf type: count}; asserts the expected 345 tensors were all replaced."""
    import gguf, torch
    if check_sha:
        got = sha256(blob)
        assert got == GGUF_SHA256, f"{blob} is not the official gemma4:e4b-it-qat GGUF (sha256 {got[:12]})"
    params = dict(model.named_parameters())
    reader = gguf.GGUFReader(str(blob))
    done = {}
    for t in reader.tensors:
        name = hf_name(t.name)
        if name is None:
            continue
        tname = t.tensor_type.name
        assert tname in EXPECTED, f"{t.name}: unexpected type {tname}"
        if name not in params and not strict:
            continue
        p = params[name]
        rows, cols = int(t.shape[1]), int(t.shape[0])          # ggml ne order is (cols, rows)
        assert tuple(p.shape) == (rows, cols), f"{t.name} -> {name}: GGUF {(rows, cols)} vs model {tuple(p.shape)}"
        data = np.asarray(t.data)
        with torch.no_grad():
            for i in range(0, rows, rows_per_chunk):         # chunked: per_layer_token_embd is 2.8B weights
                w = gguf.quants.dequantize(data[i:i + rows_per_chunk], t.tensor_type).reshape(-1, cols)
                p[i:i + w.shape[0]].copy_(torch.from_numpy(np.ascontiguousarray(w)).to(device=p.device, dtype=p.dtype))
        done[tname] = done.get(tname, 0) + 1
    if strict:
        assert done == EXPECTED, f"replaced {done}, expected {EXPECTED}"
    return done


def official_gguf():
    """The official GGUF on the volume (downloaded once from the Ollama registry), sha256-checked."""
    import urllib.request
    path = pathlib.Path(f"{MODELS}/gguf/gemma4-e4b-it-qat.gguf")
    if not path.exists() or sha256(path) != GGUF_SHA256:
        path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(GGUF_URL, tmp); assert sha256(tmp) == GGUF_SHA256, "GGUF download sha mismatch"
        tmp.rename(path); volume.commit()
    return path


def load(base, device):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
    repo, rev = BASES[base]
    path = snapshot_download(repo, revision=rev)
    model = AutoModelForMultimodalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": device})
    model.config.use_cache = False
    for q in model.parameters():
        q.requires_grad = False
    if EXACT_Q4:
        print('exact Q4 base:', apply_exact_base(model, official_gguf(), check_sha=False), flush=True)  # sha checked in official_gguf
    return model, AutoTokenizer.from_pretrained(path), path


def add_lora(model, cfg):
    from peft import LoraConfig, get_peft_model
    names = [n for n, m in model.named_modules() if is_target(n, cfg["targets"]) and hasattr(m, "weight")]
    assert names, "no target layers found"
    return get_peft_model(model, LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=0.0, target_modules=names, task_type="CAUSAL_LM"))


def check_lora(model, cfg, base_path):
    """The real trainable count must match the estimate from the raw config.json within 1%, and every adapted layer must be a text layer."""
    wrapped = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    assert wrapped and all("language_model" in n for n in wrapped), "LoRA must stay in the text layers"
    assert all("lora_" in n for n, q in model.named_parameters() if q.requires_grad)
    n = sum(q.numel() for q in model.parameters() if q.requires_grad)
    raw = json.load(open(pathlib.Path(base_path) / "config.json"))
    est = lora_params(raw.get("text_config", raw), cfg["rank"], cfg["targets"])
    print(f"LoRA wraps {len(wrapped)} layers; trainable {n:,}; estimate from config {est:,}", flush=True)
    assert abs(n - est) <= 0.01 * est, "trainable parameter count does not match the config estimate"
    return n, len(wrapped)


def save_run(model, base_path, out, cfg):
    """Adapter alone, then merged BF16 weights with Google's own tokenizer and template files, with a check that weight names match the base."""
    from safetensors import safe_open
    out = pathlib.Path(out)
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"):
        src = pathlib.Path(base_path) / name
        if src.exists():
            shutil.copy(src, out / "merged" / name)

    def tensors(folder):
        found = {}
        for f in pathlib.Path(folder).glob("*.safetensors"):
            with safe_open(f, "pt") as st:
                for key in st.keys():
                    found[key] = (f, st.get_slice(key).get_shape())
        return found

    old, new = tensors(base_path), tensors(out / "merged")
    report = {"base_tensors": len(old), "merged_tensors": len(new), "missing": sorted(set(old) - set(new))[:20],
              "unexpected": sorted(set(new) - set(old))[:20], "shape_mismatch": [k for k in set(old) & set(new) if old[k][1] != new[k][1]][:20]}
    changed = 0
    for key in set(old) & set(new):
        if is_target(key.rsplit(".weight", 1)[0], cfg["targets"]):
            with safe_open(old[key][0], "pt") as a, safe_open(new[key][0], "pt") as b:
                changed += int(not a.get_tensor(key).equal(b.get_tensor(key)))
    report["target_weights_changed"] = changed
    print("merge report:", json.dumps(report), flush=True)
    assert not report["missing"] and not report["unexpected"] and not report["shape_mismatch"], "weight names differ from the base"
    assert changed > 0, "merged weights are identical to the base, so training had no effect"
    return report


def prepare(tok, rows, max_tokens):
    enc, dropped = [], 0
    for r in rows:
        p, a = encode(tok, r["messages"], r.get("thinking", True))   # round-2 rows carry "thinking": false (H2 evaluates thinking off)
        if len(p) + len(a) > max_tokens:
            dropped += 1
            continue
        enc.append((r["task"], p, a))
    print(f"{len(enc)} examples encoded, {dropped} dropped for exceeding {max_tokens} tokens", flush=True)
    return enc


@app.function(volumes={MODELS: volume}, cpu=8, memory=65536, timeout=3600)
def preflight(base, cfg, data):
    import torch
    rows = [json.loads(l) for l in data.splitlines()]
    model, tok, path = load(base, "cpu")
    volume.commit()  # keep the download
    enc = prepare(tok, rows[:40], 1 << 30)
    lengths = sorted(len(p) + len(a) for _, p, a in enc)
    print(f"token lengths of the first {len(enc)}: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}", flush=True)

    # Reasoning must survive the template and land inside the graded part. SERA's score fell from 41% to 23% without it.
    msgs = [dict(m) for m in rows[0]["messages"]]
    msgs[-1]["reasoning"] = "SENTINEL-REASONING"
    p, a = encode(tok, msgs)
    graded = tok.decode(a)
    assert "<|channel>thought\nSENTINEL-REASONING" in graded and "SENTINEL" not in tok.decode(p), graded[:200]
    print("reasoning is rendered and graded: ok. Graded text starts:", repr(graded[:60]), flush=True)
    think = rows[0].get("thinking", True)
    p, _ = encode(tok, rows[0]["messages"], think)
    assert ("<|think|>" in tok.decode(p)) == think, f"thinking={think} is not what the rendered prompt shows"
    print(f"prompt rendered with thinking={think}: ok", flush=True)
    # compare with the server: sha256 of json.dumps(/tokenize tokens) for the same row must match (clanker's render check)
    print(f"row 0 prompt: {len(p)} tokens, ids sha256 {__import__('hashlib').sha256(json.dumps(p).encode()).hexdigest()}", flush=True)

    model = add_lora(model, cfg)
    n, layers = check_lora(model, cfg, path)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    p, a = enc[0][1][-64:], enc[0][2][:32]
    lr = 1e-4
    assert lr > 0
    optim = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr)
    t0 = time.time()
    loss = answer_loss(model, p, a, "cpu")
    loss.backward()
    grads = [q.grad.abs().sum().item() for nme, q in model.named_parameters() if "lora_B" in nme and q.grad is not None]
    assert grads and sum(grads) > 0, "no gradient reached the LoRA layers"
    optim.step()
    print(f"one training step: loss {loss.item():.3f}, {time.time() - t0:.0f}s on CPU, {len(grads)} LoRA layers got gradient", flush=True)
    save_run(model, path, "/tmp/preflight", cfg)
    print(f"PREFLIGHT PASSED base={base} config={cfg} trainable={n:,} layers={layers}", flush=True)


def run_training(base, cfg, data, epochs, lr, name, accum, max_tokens, val_every_task, val_data=""):
    import math, random
    import torch
    device = "cuda"
    assert lr > 0, "learning rate must be positive"
    rows = [json.loads(l) for l in data.splitlines()]
    model, tok, path = load(base, device)
    enc = prepare(tok, rows, max_tokens)
    tasks = sorted({e[0] for e in enc})
    if val_data:  # an explicit validation file (Aarav's split): nothing is held out of the training rows
        held = set()
        val = prepare(tok, [json.loads(l) for l in val_data.splitlines()], max_tokens)
        assert not {e[0] for e in val} & set(tasks), "a validation task is also in the training rows"
    else:
        held = set(tasks[::val_every_task]) if val_every_task else set()
        val = [e for e in enc if e[0] in held]
        enc = [e for e in enc if e[0] not in held]
    print(f"train examples {len(enc)}, validation examples {len(val)} from {len(held)} held-out tasks", flush=True)

    model = add_lora(model, cfg)
    n, layers = check_lora(model, cfg, path)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def val_loss():
        model.eval()
        with torch.no_grad():
            losses = [answer_loss(model, p, a, device).item() for _, p, a in val]
        model.train()
        return round(sum(losses) / len(losses), 4) if losses else None

    before = val_loss()
    print("validation loss before:", before, flush=True)
    random.seed(0)
    order = [e for _ in range(epochs) for e in random.sample(enc, len(enc))]
    params = [q for q in model.parameters() if q.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = math.ceil(len(order) / accum)
    warmup = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, (s + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    print(f"{len(order)} examples over {epochs} epochs, {steps} optimizer steps (accum {accum}), lr {lr}, {n:,} trainable parameters", flush=True)

    model.train()
    history, t_start = [], time.time()
    for i, (task, p, a) in enumerate(order, 1):
        t0 = time.time()
        loss = answer_loss(model, p, a, device)
        (loss / accum).backward()
        if i % accum == 0 or i == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step(); optim.zero_grad(set_to_none=True); sched.step()
        row = {"i": i, "task": task, "tokens": len(p) + len(a), "loss": round(loss.item(), 4), "s": round(time.time() - t0, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "lr": sched.get_last_lr()[0]}
        history.append(row)
        if i % 10 == 0 or i == len(order):
            print(json.dumps(row), flush=True)
        if i % 100 == 0 and val:
            print(json.dumps({"i": i, "val_loss": val_loss()}), flush=True)
        if i % len(enc) == 0 and i < len(order):  # end of an epoch (not the last): keep its adapter, so the epoch can be chosen on val
            ep = i // len(enc)
            model.save_pretrained(f"{RUNS}/{name}/adapter_e{ep}")
            volume.commit()
            print(json.dumps({"epoch": ep, "saved": f"runs/{name}/adapter_e{ep}", "val_loss": val_loss() if val else None}), flush=True)

    after = val_loss()
    print("validation loss after:", after, flush=True)
    out = pathlib.Path(f"{RUNS}/{name}")
    out.mkdir(parents=True, exist_ok=True)
    merge = save_run(model, path, out, cfg)
    result = {"name": name, "base": base, "config": cfg, "lr": lr, "epochs": epochs, "accum": accum, "max_tokens": max_tokens,
              "examples": len(enc), "val_examples": len(val), "trainable": n, "layers": layers, "val_before": before, "val_after": after,
              "peak_gb": history[-1]["peak_gb"], "total_s": round(time.time() - t_start), "merge": merge,
              "teachers": dict(__import__("collections").Counter(r.get("teacher") for r in rows)),
              "near_test_rows": sum(bool(r.get("near_test")) for r in rows), "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    volume.commit()
    print(f"TRAINING FINISHED {name}: val {before} -> {after}, peak {result['peak_gb']} GB, {result['total_s']} s", flush=True)


def file_hashes(folder):
    """{file name: (sha256, size)} for every regular file in a folder, the format of the release's MERGED_SHA256SUMS."""
    import hashlib
    out = {}
    for f in sorted(pathlib.Path(folder).iterdir()):
        if f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            out[f.name] = (h.hexdigest(), f.stat().st_size)
    return out


def parse_sums(text):
    """'<sha256>  <name>  <size>' per line -> {name: (sha256, size)}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out[parts[1]] = (parts[0], int(parts[2]))
    return out


def compare_hashes(ours, ref):
    return {"same": sorted(k for k in ours if k in ref and ours[k] == ref[k]),
            "different": sorted(k for k in ours if k in ref and ours[k] != ref[k]),
            "only_ours": sorted(k for k in ours if k not in ref), "only_ref": sorted(k for k in ref if k not in ours)}


@app.function(gpu="A100-80GB", volumes={MODELS: volume}, cpu=8, memory=65536, timeout=3600)
def merge(base, cfg, name, adapter_subdir="adapter"):
    """Re-create runs/<name>/merged from runs/<name>/<adapter_subdir> on the same GPU class, image and code path as training
    (load bf16 base at the pinned revision, PeftModel.from_pretrained, merge_and_unload, save_pretrained), so a candidate trained in
    one workspace can be reproduced in another and compared by hash. Prints the merge report and the per-file hashes."""
    from peft import PeftModel
    model, tok, path = load(base, "cuda")
    out = pathlib.Path(RUNS) / name
    peft_model = PeftModel.from_pretrained(model, str(out / adapter_subdir))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    for fname in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"):
        src = pathlib.Path(path) / fname
        if src.exists():
            shutil.copy(src, out / "merged" / fname)
    volume.commit()
    hashes = file_hashes(out / "merged")
    print("MERGED_HASHES " + json.dumps(hashes), flush=True)
    return hashes


# TRAIN_GPU (read when the app is deployed) picks the training GPU; peak memory is ~32 GB, so H100, A100-80GB or L40S all fit.
@app.function(gpu=os.environ.get("TRAIN_GPU", "A100-80GB"), volumes={MODELS: volume}, cpu=8, memory=65536, timeout=5 * 3600)
def train(base, cfg, data, epochs, lr, name, accum, max_tokens, val_every_task, val_data=""):
    run_training(base, cfg, data, epochs, lr, name, accum, max_tokens, val_every_task, val_data)


def require_workspace(workspace, env=None):
    """Opt-in, see the module docstring. Without --workspace nothing is checked, so earlier commands run as they did."""
    if not workspace:
        return None
    profile = (os.environ if env is None else env).get("MODAL_PROFILE", "")
    if not profile or profile == "deepseek":
        raise SystemExit(f"--workspace {workspace!r}: MODAL_PROFILE must name it, and may not be unset or 'deepseek' (now {profile!r})")
    if profile != workspace:
        raise SystemExit(f"--workspace {workspace!r} is not MODAL_PROFILE {profile!r}")
    return workspace


@app.local_entrypoint()
def main(step: str = "preflight", base: str = "e4b", config: str = "r64", data: str = "results/train.jsonl", name: str = "",
         epochs: int = 3, lr: float = 1e-4, accum: int = 4, max_tokens: int = 24576, val_every_task: int = 20,
         teachers: str = "", reasoning: str = "any", near_test: str = "drop", call_id: str = "", adapter: str = "adapter", ref_sums: str = "", workspace: str = "", val: str = ""):
    require_workspace(workspace)
    if step == "merge":  # spawn on the deployed app, like launch; compare with --step merge-status
        assert name, "--name is required"
        fc = modal.Function.from_name(app.name, "merge").spawn(base, CONFIGS[config], name, adapter)
        print(f"MERGE LAUNCHED {name} as {fc.object_id}; check with --step merge-status --call-id {fc.object_id} --ref-sums MERGED_SHA256SUMS", flush=True)
        return
    if step == "merge-status":
        fc = modal.FunctionCall.from_id(call_id)
        try:
            hashes = fc.get(timeout=0)
        except TimeoutError:
            print("still running:", call_id); return
        hashes = {k: tuple(v) for k, v in hashes.items()}
        print("merged files:", json.dumps({k: [v[0][:16], v[1]] for k, v in hashes.items()}))
        if ref_sums:
            print("vs reference:", json.dumps(compare_hashes(hashes, parse_sums(pathlib.Path(ref_sums).read_text()))))
        return
    if step == "status":
        fc = modal.FunctionCall.from_id(call_id)
        try:
            print("finished:", fc.get(timeout=0))
        except TimeoutError:
            print("still running:", call_id)
        return
    cfg = CONFIGS[config]
    rows = [json.loads(l) for l in pathlib.Path(data).read_text().splitlines()]
    rows = select_examples(rows, set(teachers.split(",")) if teachers else None, {"any": None, "yes": True, "no": False}[reasoning], near_test == "keep")
    print(f"{len(rows)} examples selected from {data} (teachers={teachers or 'all'}, reasoning={reasoning}, near_test={near_test})", flush=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    val_text = pathlib.Path(val).read_text() if val else ""
    if val:
        print(f"validation rows from {val}: {len(val_text.splitlines())}", flush=True)
    if step == "preflight":
        preflight.remote(base, cfg, text)
    elif step == "train":
        assert name, "--name is required"
        train.remote(base, cfg, text, epochs, lr, name, accum, max_tokens, val_every_task, val_text)
    elif step == "launch":
        assert name, "--name is required"
        fc = modal.Function.from_name(app.name, "train").spawn(base, cfg, text, epochs, lr, name, accum, max_tokens, val_every_task, val_text)
        print(f"LAUNCHED {name} as {fc.object_id}; check with --step status --call-id {fc.object_id}", flush=True)
    else:
        raise SystemExit(f"unknown step {step}")
