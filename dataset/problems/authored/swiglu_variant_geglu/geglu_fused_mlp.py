"""GeGLU MLP: x @ [W_gate | W_up] (fused) -> gelu_tanh(gate) * up -> @ W_down."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'geglu_fused_mlp',
    'batch': 2,
    'seq_len': 2048,
    'd_model': 2048,
    'd_ff': 16384,
    'weight_scale': 0.02,
    'gelu_approximate': True,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kgu, kd = jax.random.split(key, 3)
    B, S, D, F = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model'], CONFIG['d_ff']
    x = jax.random.normal(kx, (B, S, D), dtype)
    w_gate_up = jax.random.normal(kgu, (D, 2 * F), dtype) * CONFIG['weight_scale']
    w_down = jax.random.normal(kd, (F, D), dtype) * CONFIG['weight_scale']
    return x, w_gate_up, w_down


def workload(x, w_gate_up, w_down):
    F = w_down.shape[0]
    # Fused gate/up projection with f32 accumulation.
    gu = jnp.einsum('bsd,df->bsf', x, w_gate_up,
                    preferred_element_type=jnp.float32)
    gate = gu[..., :F]
    up = gu[..., F:]
    h = jax.nn.gelu(gate, approximate=CONFIG['gelu_approximate']) * up
    h = h.astype(x.dtype)
    # Down projection with f32 accumulation.
    out = jnp.einsum('bsf,fd->bsd', h, w_down,
                     preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
