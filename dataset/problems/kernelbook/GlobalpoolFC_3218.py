"""[bf16 variant] GlobalpoolFC: global average pool over NCHW spatial axes, flatten, linear layer with bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GlobalpoolFC_3218',
    'batch': 8,
    'num_in': 128,
    'height': 256,
    'width': 256,
    'num_class': 512,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key_x, key_w, key_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        key_x,
        (CONFIG['batch'], CONFIG['num_in'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(key_w, (CONFIG['num_in'], CONFIG['num_class']), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(key_b, (CONFIG['num_class'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    y = jnp.mean(x, axis=(2, 3), keepdims=True)
    y = y.reshape(y.shape[0], -1)
    return y @ w + b
