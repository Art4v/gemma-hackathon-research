"""[bf16 variant] Fusion3_MinusFCLayer: concat(x1, x2, x3, x1-x2, x1-x3, x2-x3) -> Linear(6D -> D)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Fusion3_MinusFCLayer_17850',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'input_dim': 512,
    'num_fused': 6,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, kw, kb = jax.random.split(jax.random.key(0), 5)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_dim'])
    d = CONFIG['input_dim']
    x1 = jax.random.uniform(k1, shape, dtype)
    x2 = jax.random.uniform(k2, shape, dtype)
    x3 = jax.random.uniform(k3, shape, dtype)
    w = jax.random.normal(kw, (CONFIG['num_fused'] * d, d), dtype) * 0.02
    b = jax.random.normal(kb, (d,), dtype) * 0.02
    return x1, x2, x3, w, b


def workload(x1, x2, x3, w, b):
    fused = jnp.concatenate([x1, x2, x3, x1 - x2, x1 - x3, x2 - x3], axis=-1)
    return fused @ w + b
