"""GeGLU MLP: x @ W_gate_up (fused) -> split -> gelu_tanh(gate) * up -> @ W_down."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'geglu_mlp_fused_gate_up',
    'batch': 2,
    'seq_len': 2048,
    'd_model': 2048,
    'd_ff': 16384,
    'weight_scale': 0.02,
    'gelu_approximate': True,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kgu, kd = jax.random.split(key, 3)
    tokens = CONFIG['batch'] * CONFIG['seq_len']
    d_model = CONFIG['d_model']
    d_ff = CONFIG['d_ff']
    scale = CONFIG['weight_scale']
    x = jax.random.normal(kx, (tokens, d_model), dtype=dtype)
    w_gate_up = jax.random.normal(kgu, (d_model, 2 * d_ff), dtype=dtype) * scale
    w_down = jax.random.normal(kd, (d_ff, d_model), dtype=dtype) * scale
    return x, w_gate_up, w_down


def workload(x, w_gate_up, w_down):
    d_ff = w_down.shape[0]
    h = jnp.dot(x, w_gate_up, preferred_element_type=jnp.float32)
    gate = h[:, :d_ff]
    up = h[:, d_ff:]
    act = jax.nn.gelu(gate, approximate=CONFIG['gelu_approximate']) * up
    act = act.astype(x.dtype)
    y = jnp.dot(act, w_down, preferred_element_type=jnp.float32)
    return y.astype(x.dtype)
