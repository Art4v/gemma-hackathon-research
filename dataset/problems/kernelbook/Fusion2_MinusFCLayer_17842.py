"""Fusion2_MinusFCLayer: concat([x1, x2, x1 - x2], -1) -> Linear(3D -> D)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Fusion2_MinusFCLayer_17842',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'input_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_dim'])
    d = CONFIG['input_dim']
    input1 = jax.random.uniform(k1, shape, dtype)
    input2 = jax.random.uniform(k2, shape, dtype)
    w = jax.random.normal(k3, (3 * d, d), dtype) * 0.02
    b = jnp.zeros((d,), dtype)
    return input1, input2, w, b


def workload(input1, input2, w, b):
    cat = jnp.concatenate([input1, input2, input1 - input2], axis=-1)
    return cat @ w + b
