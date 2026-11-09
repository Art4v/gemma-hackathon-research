"""Variance over channel dim: mean -> sub -> square -> mean (keepdim, population variance)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'VarianceC_4476',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'axis': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    axis = CONFIG['axis']
    mean_x = jnp.mean(x, axis=axis, keepdims=True)
    sub_x = x - mean_x
    return jnp.mean(sub_x * sub_x, axis=axis, keepdims=True)
