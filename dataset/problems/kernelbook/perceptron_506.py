"""[bf16 variant] Linear projection on the last axis (x @ w + b) followed by softmax over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'perceptron_506',
    'batch': 8,
    'dim1': 256,
    'dim2': 256,
    'n_channels': 128,
    'out_features': 10,
    'softmax_axis': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key_x, key_w, key_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        key_x,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_channels']),
        dtype,
    )
    w = jax.random.normal(
        key_w, (CONFIG['n_channels'], CONFIG['out_features']), dtype
    ) * CONFIG['weight_scale']
    b = jax.random.normal(key_b, (CONFIG['out_features'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    y = x @ w + b
    return jax.nn.softmax(y, axis=CONFIG['softmax_axis'])
