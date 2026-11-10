"""Quantization-aware base for LoRA training: overwrite the QAT checkpoint's weights with the exact values Ollama serves.

The served model is the official GGUF (ollama `gemma4:e4b-it-qat`, blob sha256 e8b6a059...). Its weights differ from the bf16 QAT
checkpoint (google/gemma-4-E4B-it-qat-q4_0-unquantized @ 476025a) by the 4-bit rounding. Training the LoRA on the bf16 weights teaches
corrections for weights the served model never runs; training it on the dequantized GGUF weights teaches corrections for the real ones.

Replaced (345 tensors, all checked by name and shape):
  342 Q4_0  blk.N.{attn_q,attn_k,attn_v,attn_output,ffn_gate,ffn_up,ffn_down,inp_gate,proj}  ->  layers.N.{self_attn.*,mlp.*,per_layer_*}
    2 Q6_K  token_embd (tied to the output head), per_layer_token_embd
    1 F16   per_layer_model_proj
Dequantized with gguf-py's reference (for Q4_0 exactly d*(q-8)), then stored in the model's bf16: the bf16 rounding is at most 2.7% of
one 4-bit step (measured on blk.0.attn_k), against up to 50% of a step for the 4-bit rounding being modelled. The 321 F32 tensors
(norms, scalars) are bf16 values in the checkpoint already and are left as loaded.
The served adapter is folded the same way (fold_local.py: dequant(Q4_0) + delta, stored at 8 bits), so training and serving share one base.
"""
import hashlib, re
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
