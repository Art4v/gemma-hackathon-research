"""Batched MoE expert FFN on a fixed-capacity buffer: [E,C,d] x [E,d,f] + b1 -> GELU -> [E,C,f] x [E,f,d] + b2."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'moe_expert_ffn_capacity_gelu',
    'num_experts': 16,
    'capacity': 512,
    'd_model': 1024,
    'd_ff': 4096,
    'weight_scale': 0.02,
    'gelu_approximate': True,
}


def create_inputs(dtype=jnp.float32):
    E = CONFIG['num_experts']
    C = CONFIG['capacity']
    d = CONFIG['d_model']
    f = CONFIG['d_ff']
    s = CONFIG['weight_scale']
    key = jax.random.key(0)
    kx, k1, kb1, k2, kb2 = jax.random.split(key, 5)
    x = jax.random.normal(kx, (E, C, d), dtype)
    w1 = jax.random.normal(k1, (E, d, f), dtype) * s
    b1 = jax.random.normal(kb1, (E, f), dtype) * s
    w2 = jax.random.normal(k2, (E, f, d), dtype) * s
    b2 = jax.random.normal(kb2, (E, d), dtype) * s
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    out_dtype = x.dtype
    # Up-projection per expert, f32 accumulation.
    h = jnp.einsum('ecd,edf->ecf', x, w1, preferred_element_type=jnp.float32)
    h = h + b1.astype(jnp.float32)[:, None, :]
    h = jax.nn.gelu(h, approximate=CONFIG['gelu_approximate'])
    h = h.astype(out_dtype)
    # Down-projection per expert, f32 accumulation.
    y = jnp.einsum('ecf,efd->ecd', h, w2, preferred_element_type=jnp.float32)
    y = y + b2.astype(jnp.float32)[:, None, :]
    return y.astype(out_dtype)
