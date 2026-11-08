"""GeGLU MLP: x @ W_gate_up (fused) -> split -> gelu_tanh(gate) * up -> @ W_down (bf16, f32 accumulation)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'geglu_fused_gate_up_mlp',
    'batch': 2,
    'seq_len': 2048,
    'd_model': 3072,
    'd_ff': 24576,
    'gelu_approximate': True,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kgu, kd = jax.random.split(key, 3)
    T = CONFIG['batch'] * CONFIG['seq_len']
    D = CONFIG['d_model']
    F = CONFIG['d_ff']
    x = jax.random.normal(kx, (T, D), dtype)
    w_gate_up = jax.random.normal(kgu, (D, 2 * F), dtype) * 0.02
    w_down = jax.random.normal(kd, (F, D), dtype) * 0.02
    return x, w_gate_up, w_down


def workload(x, w_gate_up, w_down):
    F = w_down.shape[0]
    # Fused gate/up projection with f32 accumulation.
    h = jnp.dot(x, w_gate_up, preferred_element_type=jnp.float32)
    gate = h[:, :F]
    up = h[:, F:]
    # Tanh-approximate GELU (Gemma's gelu_pytorch_tanh) in f32.
    act = jax.nn.gelu(gate, approximate=CONFIG['gelu_approximate']) * up
    act = act.astype(x.dtype)
    # Down projection with f32 accumulation.
    out = jnp.dot(act, w_down, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
