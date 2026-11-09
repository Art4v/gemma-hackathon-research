"""GeneralRelu: relu (or leaky_relu) -> optional subtract -> optional clamp_max."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeneralRelu_756',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'leak': None,
    'sub': None,
    'maxv': None,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    leak, sub, maxv = CONFIG['leak'], CONFIG['sub'], CONFIG['maxv']
    if leak is not None:
        x = jnp.where(x >= 0, x, x * leak)
    else:
        x = jax.nn.relu(x)
    if sub is not None:
        x = x - sub
    if maxv is not None:
        x = jnp.minimum(x, maxv)
    return x
