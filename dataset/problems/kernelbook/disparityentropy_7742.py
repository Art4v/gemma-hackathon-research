"""Disparity entropy: element-wise -x * log(x) followed by a sum over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'disparityentropy_7742',
    'batch': 8,
    'maxdisp': 128,
    'height': 256,
    'width': 256,
    'sum_axis': 1,
    'minval': 1e-6,
    'maxval': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['maxdisp'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype, minval=CONFIG['minval'], maxval=CONFIG['maxval'])
    return (x,)


def workload(x):
    return jnp.sum(-x * jnp.log(x), axis=CONFIG['sum_axis'])
