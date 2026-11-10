"""[bf16 variant] Linear (x @ w + b) followed by softmax over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinModel_17578',
    'batch': 8,
    'dim_1': 128,
    'dim_2': 128,
    'in_dim': 256,
    'out_dim': 256,
    'weight_scale': 0.02,
    'softmax_axis': -1,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['dim_1'], CONFIG['dim_2'], CONFIG['in_dim']),
        dtype,
    )
    w = jax.random.normal(kw, (CONFIG['in_dim'], CONFIG['out_dim']), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(kb, (CONFIG['out_dim'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    out = x @ w + b
    return jax.nn.softmax(out, axis=CONFIG['softmax_axis'])
