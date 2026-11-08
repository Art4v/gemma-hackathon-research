"""Batched MoE expert FFN on fixed-capacity buffer: [E,C,d] x [E,d,f] -> GELU -> [E,C,f] x [E,f,d]."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'moe_capacity_expert_gelu_ffn',
    'num_experts': 8,
    'capacity': 1024,
    'd_model': 2048,
    'd_ff': 5632,
    'gelu_approximate': True,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    E = CONFIG['num_experts']
    C = CONFIG['capacity']
    d = CONFIG['d_model']
    f = CONFIG['d_ff']
    key = jax.random.key(0)
    kx, k1, k2 = jax.random.split(key, 3)
    x = jax.random.normal(kx, (E, C, d), dtype)
    w1 = jax.random.normal(k1, (E, d, f), dtype) * CONFIG['weight_scale']
    w2 = jax.random.normal(k2, (E, f, d), dtype) * CONFIG['weight_scale']
    return x, w1, w2


def workload(x, w1, w2):
    out_dtype = x.dtype
    # Up-projection per expert, f32 accumulation.
    h = jnp.einsum('ecd,edf->ecf', x, w1, preferred_element_type=jnp.float32)
    h = jax.nn.gelu(h, approximate=CONFIG['gelu_approximate'])
    h = h.astype(out_dtype)
    # Down-projection per expert, f32 accumulation.
    y = jnp.einsum('ecf,efd->ecd', h, w2, preferred_element_type=jnp.float32)
    return y.astype(out_dtype)
