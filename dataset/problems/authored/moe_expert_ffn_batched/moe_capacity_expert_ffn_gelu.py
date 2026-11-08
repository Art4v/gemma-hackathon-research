"""Capacity-buffered MoE expert FFN: batched [E,C,d]x[E,d,f] + bias -> GELU(tanh) -> [E,C,f]x[E,f,d] + bias -> capacity mask."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'moe_capacity_expert_ffn_gelu',
    'num_experts': 8,
    'capacity': 512,
    'd_model': 2048,
    'd_ff': 6144,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    E = CONFIG['num_experts']
    C = CONFIG['capacity']
    d = CONFIG['d_model']
    f = CONFIG['d_ff']
    s = CONFIG['weight_scale']
    key = jax.random.key(0)
    kx, k1, kb1, k2, kb2, kc = jax.random.split(key, 6)
    x = jax.random.normal(kx, (E, C, d), dtype)
    w1 = jax.random.normal(k1, (E, d, f), dtype) * s
    b1 = jax.random.normal(kb1, (E, f), dtype) * s
    w2 = jax.random.normal(k2, (E, f, d), dtype) * s
    b2 = jax.random.normal(kb2, (E, d), dtype) * s
    counts = jax.random.randint(kc, (E,), 0, C + 1, dtype=jnp.int32)
    return x, w1, b1, w2, b2, counts


def workload(x, w1, b1, w2, b2, counts):
    E, C, _ = x.shape
    h = jnp.einsum('ecd,edf->ecf', x, w1, preferred_element_type=jnp.float32)
    h = h + b1.astype(jnp.float32)[:, None, :]
    h = jax.nn.gelu(h, approximate=True).astype(x.dtype)
    y = jnp.einsum('ecf,efd->ecd', h, w2, preferred_element_type=jnp.float32)
    y = y + b2.astype(jnp.float32)[:, None, :]
    slot = jnp.arange(C, dtype=jnp.int32)[None, :]
    valid = slot < counts[:, None]
    y = jnp.where(valid[:, :, None], y, 0.0)
    return y.astype(x.dtype)
