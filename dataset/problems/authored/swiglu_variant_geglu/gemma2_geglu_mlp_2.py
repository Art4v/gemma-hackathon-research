"""GeGLU MLP block with fused gate-up projection and float32 accumulation (Gemma-2-9B)."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_geglu_mlp',
    'batch': 4,
    'seq_len': 2048,
    'emb_dim': 3584,
    'mlp_dim': 14336,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    B, S, D, F = (
        CONFIG['batch'],
        CONFIG['seq_len'],
        CONFIG['emb_dim'],
        CONFIG['mlp_dim'],
    )
    x = jax.random.normal(k1, (B, S, D), dtype=dtype)
    w_gate_up = jax.random.normal(k2, (D, 2 * F), dtype=dtype) * 0.02
    w_down = jax.random.normal(k3, (F, D), dtype=dtype) * 0.02
    return x, w_gate_up, w_down


def workload(x, w_gate_up, w_down):
    F = CONFIG['mlp_dim']
    gate_up = jnp.dot(x, w_gate_up, preferred_element_type=jnp.float32)
    gate = gate_up[..., :F]
    up = gate_up[..., F:]
    hidden = (jax.nn.gelu(gate, approximate=True) * up).astype(x.dtype)
    out = jnp.dot(hidden, w_down, preferred_element_type=jnp.float32).astype(x.dtype)
    return out
