"""Fused GeGLU MLP block with concatenated gate-up projection and float32 accumulation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_geglu_mlp',
    'batch_size': 4,
    'seq_len': 2048,
    'hidden_size': 3584,
    'intermediate_size': 14336,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    H = CONFIG['hidden_size']
    I = CONFIG['intermediate_size']

    x = jax.random.normal(k1, (B, S, H), dtype=dtype)
    w_gate_up = jax.random.normal(k2, (H, 2 * I), dtype=dtype) * 0.02
    w_down = jax.random.normal(k3, (I, H), dtype=dtype) * 0.02
    return x, w_gate_up, w_down

def workload(x, w_gate_up, w_down):
    I = CONFIG['intermediate_size']
    orig_dtype = x.dtype

    gate_up = jnp.dot(x, w_gate_up, preferred_element_type=jnp.float32)
    gate = gate_up[..., :I]
    up = gate_up[..., I:]

    hidden = (jax.nn.gelu(gate, approximate=True) * up).astype(orig_dtype)
    out = jnp.dot(hidden, w_down, preferred_element_type=jnp.float32).astype(orig_dtype)
    return out
