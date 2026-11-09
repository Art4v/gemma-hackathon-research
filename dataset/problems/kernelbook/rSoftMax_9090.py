"""[bf16 variant] rSoftMax: view(B, card, radix, -1) -> transpose(1, 2) -> softmax over radix -> reshape(B, -1) (sigmoid if radix == 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'rSoftMax_9090',
    'batch': 64,
    'channels': 1024,
    'height': 32,
    'width': 32,
    'radix': 4,
    'cardinality': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    batch = x.shape[0]
    radix = CONFIG['radix']
    cardinality = CONFIG['cardinality']
    if radix > 1:
        y = x.reshape(batch, cardinality, radix, -1)
        y = jnp.transpose(y, (0, 2, 1, 3))
        y = jax.nn.softmax(y, axis=1)
        return y.reshape(batch, -1)
    return jax.nn.sigmoid(x)
