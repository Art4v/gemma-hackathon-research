"""[bf16 variant] rSoftMax: reshape to (batch, cardinality, radix, -1), swap axes 1 and 2, softmax over radix axis, flatten (sigmoid if radix == 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'rSoftMax_17292',
    'batch': 8,
    'channels': 256,
    'height': 256,
    'width': 128,
    'radix': 4,
    'cardinality': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    batch = x.shape[0]
    if CONFIG['radix'] > 1:
        y = jnp.reshape(x, (batch, CONFIG['cardinality'], CONFIG['radix'], -1))
        y = jnp.swapaxes(y, 1, 2)
        y = jax.nn.softmax(y, axis=1)
        return jnp.reshape(y, (batch, -1))
    return jax.nn.sigmoid(x)
