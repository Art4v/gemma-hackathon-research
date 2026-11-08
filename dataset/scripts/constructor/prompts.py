"""Every prompt the constructor agent sees. Versioned: PROMPT_VERSION is recorded on each candidate and problem record.

Three jobs, all producing plain-JAX reference modules: port a KernelBench PyTorch module, author a new p-style problem from a
category spec, and write a reference for a scraped human Pallas kernel. The Pallas kernel-writing prompt was archived on Nov 8
(dataset/archive/pallas/scripts/constructor/prompts.py) when kernel generation was paused.
"""
from __future__ import annotations

import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import REPO, fence, sha256  # noqa: E402

PROMPT_VERSION = "prompts-2"  # prompts-1 also carried the Pallas kernel prompt (archive/pallas)
DOCS_PATH = REPO / "pallas_guides" / "teacher_docs.md"


def docs() -> tuple:
    """(text, sha256) of the Pallas TPU docs bundle, or ('', None) if absent."""
    if DOCS_PATH.exists():
        t = DOCS_PATH.read_text()
        return t, sha256(t)
    return "", None


# ---- porting KernelBench ----

PORT_SYSTEM = """You port KernelBench problems (PyTorch nn.Module) to JAXBench-style plain-JAX baseline modules. The port is the REFERENCE a Pallas kernel will be judged against, so it must be plain JAX (no Pallas), correct, and deterministic.

Output module shape, exactly:
- Module docstring: one line naming the op chain.
- `import jax`, `import jax.numpy as jnp` (and `from jax import lax` if needed). Nothing else.
- `CONFIG = {'name': '<kernelbench stem>', ...every size...}`: every dimension, eps, scale factor and shape parameter lives here.
- `def create_inputs(dtype=jnp.float32):` returns a tuple of ALL arrays `workload` takes: activations first, then weights/biases/parameters. Activations: `jax.random.uniform` or `normal` from `jax.random.key(0)` split as needed. Weights: `jax.random.normal(k, shape, dtype) * 0.02` (like the exam's GEMM tasks). Norm scale/shift parameters: deterministic constants (`jnp.ones`, `jnp.zeros`) like the exam's norm tasks. Integer inputs (indices, lengths) from `jax.random.randint`. Deterministic: same values on every call. No Python RNG.
- `def workload(*inputs):` plain JAX, reading sizes from CONFIG or from array shapes, never from literals. Returns one array or a tuple.
- No benchmark() function, no __main__ block.

Semantics: inference mode. Dropout is the identity. BatchNorm uses BATCH statistics (JAXBench convention) with gamma=1, beta=0 unless the module trains them. nn.Linear weight is stored (out, in) in PyTorch; pass it to workload as (in, out) and use x @ w. Conv: inputs NCHW at the interface like PyTorch, convert to NHWC inside with jnp.transpose, use lax.conv_general_dilated, convert back. Match PyTorch's default eps and padding.

Shapes: scale KernelBench's small defaults up so total input bytes are between 50 MB and 600 MB (f32), with dimensions multiples of 128 where the op allows (batch at least 8, multiple of 8). Keep 3 x input bytes under 20 GB.

Reply as JSON: `plan` (which semantics choices you made and the scaled shapes), `code` (the module).
"""


def port_user(kb_stem: str, kb_source: str, retry_note: str | None = None) -> str:
    s = f"KernelBench problem `{kb_stem}`:\n\n" + fence(kb_source)
    if retry_note:
        s += "\n\nYour previous port failed the equivalence check against PyTorch:\n" + retry_note + "\n\nFix it and return the complete module."
    return s


# ---- authoring p-style problems ----

AUTHOR_SYSTEM = PORT_SYSTEM.split("Semantics:")[0].replace(
    "You port KernelBench problems (PyTorch nn.Module) to JAXBench-style plain-JAX baseline modules. The port is the REFERENCE a Pallas kernel will be judged against, so it must be plain JAX (no Pallas), correct, and deterministic.",
    "You author new model-operator problems as JAXBench-style plain-JAX baseline modules. Each is the REFERENCE a Pallas kernel will be judged against, so it must be plain JAX (no Pallas), numerically sane in bfloat16 (accumulate in float32 inside the reference where a kernel would), and deterministic.") + """
The problem must be a real operator from a modern transformer, state-space or MoE model, at realistic serving or training sizes (50 MB to 1 GB of input, bfloat16 activations unless the spec says otherwise). It must NOT be one of these exam operators or a trivial reshaping of one: flash attention (standard causal MHA), GQA attention, MLA attention, paged attention, ragged paged attention, sparse/block-sparse attention, flex attention, plain GEMM, SwiGLU MLP, sparse MoE dispatch, Megablox grouped matmul, RMSNorm alone, cross entropy alone, ragged dot, RetNet retention, Mamba2 SSD, triangle multiplication. Choose a different operator or a materially different fusion within the requested category.

Reply as JSON: `name` (a short snake_case identifier for the problem), `plan` (what the operator is, which model uses it, why it is not an exam operator, the shapes), `code` (the module; CONFIG['name'] equals `name`).
"""

AUTHOR_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "plan": {"type": "string"}, "code": {"type": "string"}},
                 "required": ["name", "plan", "code"], "additionalProperties": False}

CATEGORIES = {
    "attention_sliding_window": "Sliding-window causal attention (window << seq_len) with a fixed bf16 Q/K/V layout, softmax in f32.",
    "attention_alibi": "Multi-head attention with ALiBi positional bias (per-head slopes), causal, bf16, softmax in f32.",
    "attention_gqa_sink": "Grouped-query attention with attention sinks (learned per-head sink logits) and a sliding window; a head ratio other than 8:1.",
    "norm_residual_fusion": "Pre-norm residual block fusion: residual add, then RMSNorm or LayerNorm, returning both the normed output and the updated residual stream (bf16 in, f32 statistics).",
    "norm_gated": "Gated RMSNorm as in Mamba2/Gated DeltaNet: normalize x, multiply by silu(gate), scale by weight.",
    "moe_topk_routing": "MoE router: logits = x @ W_router, softmax, top-k selection with renormalized weights, returning expert indices and weights (k=2 or 8, E=64..256 experts).",
    "moe_expert_bincount": "Compute per-expert token counts and group offsets from routing indices, then a capacity-limited dense dispatch mask (integer-heavy with a float gate scaling).",
    "kv_cache_update": "Write new K/V for a batch of sequences into a paged KV cache at given slot indices and return the updated cache (scatter by slot mapping, bf16).",
    "kv_quantized_dequant_matmul": "Dequantize an int8 weight matrix with per-channel scales and multiply by bf16 activations, with f32 accumulation.",
    "quantized_fp8_matmul": "Blockwise-scaled fp8 (float8_e4m3fn) matmul: inputs quantized in 128x128 blocks with f32 scales; dequantize-and-matmul in the reference.",
    "ssm_scan_step": "A selective scan (Mamba-1 style) over sequence length with diagonal A, input-dependent B and C, returning the output sequence (f32 state).",
    "rope_qk_norm": "Apply QK-RMSNorm then rotary position embedding to Q and K for all heads (bf16 in, f32 math), returning rotated Q and K.",
    "swiglu_variant_geglu": "GeGLU MLP block: x @ [W_gate, W_up] fused, gelu(gate) * up, then @ W_down, bf16 with f32 accumulation.",
    "softcapped_logits": "Final-layer logits with tanh soft-capping (Gemma style) and a fused log-softmax, bf16 in, f32 out.",
    "cross_entropy_chunked_softcap": "Fused linear + softcap + cross-entropy loss returning per-token losses (not the exam's plain cross entropy; includes the projection).",
}


def author_user(category: str, spec: str, avoid: list | None = None, retry_note: str | None = None) -> str:
    s = f"Category: {category}\nSpec: {spec}\n"
    if avoid:
        s += "Problems already written in this category (choose a materially different operator or configuration): " + ", ".join(avoid) + "\n"
    if retry_note:
        s += "\nYour previous attempt was rejected:\n" + retry_note + "\n\nReturn a different problem."
    return s


# ---- wrapping scraped human kernels ----

WRAP_SYSTEM = """You turn a public Pallas TPU kernel into a dataset problem and solution pair.

Given the kernel's source, produce:
1. `ref_code`: a plain-JAX baseline module (JAXBench style: docstring, imports, CONFIG with every size, `create_inputs(dtype=...)` deterministic from jax.random.key(0) with activations random and norm parameters constant, `workload(*inputs)`) that computes exactly what the kernel computes for one realistic configuration. No Pallas in the reference.
2. `kernel_code`: a self-contained module defining `workload(*inputs)` with the same arguments, that calls the human kernel. Inline the kernel source (and only the helpers it needs); remove unrelated code paths and config knobs; keep the kernel's math unchanged. Outside pl.pallas_call only layout/dtype ops. Imports limited to jax, jax.numpy, functools, math, numpy, jax.experimental.pallas and jax.experimental.pallas.tpu.

If the kernel cannot be expressed this way (needs SparseCore, multi-device collectives, or custom calls), return `skip` = true with the reason.

Reply as JSON: `plan`, `skip` (bool), `reason` (string, empty unless skip), `ref_code`, `kernel_code`.
"""

WRAP_SCHEMA = {"type": "object", "properties": {"plan": {"type": "string"}, "skip": {"type": "boolean"}, "reason": {"type": "string"},
                                                "ref_code": {"type": "string"}, "kernel_code": {"type": "string"}},
               "required": ["plan", "skip", "reason", "ref_code", "kernel_code"], "additionalProperties": False}


def wrap_user(repo: str, path: str, source: str, licence: str) -> str:
    return f"Kernel from {repo} ({licence}), file {path}:\n\n" + fence(source)


def prompt_versions() -> dict:
    _, sha = docs()
    return {"prompt_version": PROMPT_VERSION, "docs_sha256": sha}


if __name__ == "__main__":
    print(json.dumps({**prompt_versions(), "categories": len(CATEGORIES)}))
