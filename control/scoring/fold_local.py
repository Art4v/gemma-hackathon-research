"""Fold a LoRA into a copy of the official gemma4:e4b-it-qat GGUF without 4-bit rounding, on this Mac (CPU, no Modal).

  python fold_local.py BLOB ADAPTER_DIR OUT.gguf --dtype f16|q8_0 [--zero]

For each of the 258 LoRA target tensors: new = dtype( dequant(official Q4_0) + (alpha/r) * B @ A ). The base part is the official
4-bit model's exact weights (what the bare arm runs); only the LoRA delta is added, at a precision where a delta of ~2% of a Q4 step
still exists. --zero adds no delta: the matching base model at the same precision (the bare_stack for this file).
Every other tensor (embeddings, per-layer embeddings, norms) and all metadata are copied byte-for-byte.
"""
import argparse, json, re, sys, time
import numpy as np, gguf

import os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # to_ollama.py ships next to this file (control/scoring/)
sys.path.append("/Users/hrudaynara/Projects/gemma-hackathon-research/training")  # fallback: the original location on Hruday's Mac
from to_ollama import SafeTensors, GGUF_TO_HF  # noqa: E402

TARGET = re.compile(r"blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight")
QT = gguf.GGMLQuantizationType

ap = argparse.ArgumentParser()
ap.add_argument("blob"); ap.add_argument("adapter"); ap.add_argument("out")
ap.add_argument("--dtype", choices=["f16", "q8_0"], default="f16"); ap.add_argument("--zero", action="store_true")
a = ap.parse_args()

reader = gguf.GGUFReader(a.blob)
arch = reader.fields["general.architecture"].contents()
cfg = json.load(open(f"{a.adapter}/adapter_config.json")); scale = cfg["lora_alpha"] / cfg["r"]
ad = SafeTensors(a.adapter)
writer = gguf.GGUFWriter(a.out, arch=arch, endianess=reader.endianess)
if reader.alignment != gguf.GGUF_DEFAULT_ALIGNMENT:
    writer.data_alignment = reader.alignment

for field in reader.fields.values():  # metadata, exactly as gguf_new_metadata copies it
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vt = field.types[0]
    writer.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)


def new_tensor(t):
    """(numpy array to write, ggml type) for a target tensor."""
    m = TARGET.fullmatch(t.name)
    W = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)  # (out, in): the official weights
    if not a.zero:
        hf = f"base_model.model.model.language_model.layers.{m.group(1)}.{GGUF_TO_HF[m.group(2)]}"
        W = W + scale * (ad.get(hf + ".lora_B.weight") @ ad.get(hf + ".lora_A.weight"))
    if a.dtype == "f16":
        return W.astype(np.float16), QT.F16
    return gguf.quants.quantize(W, QT.Q8_0), QT.Q8_0


targets, plan = 0, []
for t in reader.tensors:
    if TARGET.fullmatch(t.name):
        assert t.tensor_type == QT.Q4_0, (t.name, t.tensor_type)
        out_rows, in_cols = int(t.shape[1]), int(t.shape[0])  # gguf shape is (ne0=in, ne1=out)
        if a.dtype == "f16":
            writer.add_tensor_info(t.name, (out_rows, in_cols), np.dtype(np.float16), out_rows * in_cols * 2, QT.F16)
        else:
            nb = out_rows * (in_cols // 32) * 34
            writer.add_tensor_info(t.name, (out_rows, in_cols // 32 * 34), np.dtype(np.uint8), nb, QT.Q8_0)
        targets += 1
    else:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
assert targets == 258, targets

writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
t0 = time.time()
for i, t in enumerate(reader.tensors):
    if TARGET.fullmatch(t.name):
        arr, _ = new_tensor(t)
        writer.write_tensor_data(arr, tensor_endianess=reader.endianess)
    else:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
writer.close()
print(f"wrote {a.out}: {targets} target tensors as {a.dtype}{' (zero delta)' if a.zero else ''}, {time.time() - t0:.0f}s")
