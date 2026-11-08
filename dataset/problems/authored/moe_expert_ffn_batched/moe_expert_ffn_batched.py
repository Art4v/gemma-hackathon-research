"""Batched MoE expert FFN on a fixed-capacity buffer with GELU activation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'moe_expert_ffn_batched',
    'num_experts': 16,
    'capacity': 512,
    'd_model': 2048,
    'd_ff': 8192,
}

def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    E = CONFIG['num_experts']
    C = CONFIG['capacity']
    d = CONFIG['d_model']
    f = CONFIG['d_ff']
    x = jax.random.normal(k0, (E, C, d), dtype=dtype)
    w_up = jax.random.normal(k1, (E, d, f), dtype=dtype) * 0.02
    w_down = jax.random.normal(k2, (E, f, d), dtype=dtype) * 0.02
    return x, w_up, w_down

def workload(x, w_up, w_down):
    hidden = jnp.einsum('ecd,edf->ecf', x, w_up, preferred_element_type=jnp.float32)
    activated = jax.nn.gelu(hidden, approximate=True).astype(x.dtype)
    out = jnp.einsum('ecf,efd->ecd', activated, w_down, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
