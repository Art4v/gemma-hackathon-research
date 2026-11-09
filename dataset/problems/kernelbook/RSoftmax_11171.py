"""RSoftmax: view(B, groups, radix, -1) -> transpose(1, 2) -> softmax over radix -> reshape(B, -1) (sigmoid if radix == 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RSoftmax_11171',
    'batch': 32,
    'channels': 512,
    'height': 64,
    'width': 64,
    'radix': 4,
    'groups': 1,
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
    groups = CONFIG['groups']
    batch = x.shape[0]
    if radix > 1:
        y = x.reshape(batch, groups, radix, -1)
        y = jnp.transpose(y, (0, 2, 1, 3))
        y = jax.nn.softmax(y, axis=1)
        return y.reshape(batch, -1)
    return jax.nn.sigmoid(x)
