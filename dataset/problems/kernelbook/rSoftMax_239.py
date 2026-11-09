"""rSoftMax: grouped reshape -> transpose(cardinality, radix) -> softmax over radix -> flatten (sigmoid if radix == 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'rSoftMax_239',
    'batch': 128,
    'channels': 256,
    'height': 32,
    'width': 32,
    'radix': 4,
    'cardinality': 4,
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
    radix = CONFIG['radix']
    cardinality = CONFIG['cardinality']
    batch = x.shape[0]
    if radix > 1:
        y = x.reshape(batch, cardinality, radix, -1)
        y = jnp.swapaxes(y, 1, 2)
        y = jax.nn.softmax(y, axis=1)
        return y.reshape(batch, -1)
    return jax.nn.sigmoid(x)
