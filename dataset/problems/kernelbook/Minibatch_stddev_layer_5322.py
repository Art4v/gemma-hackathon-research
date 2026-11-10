"""[bf16 variant] Minibatch stddev: reshape groups, group variance, sqrt(+eps), spatial/channel mean, tile, concat as new feature channel."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Minibatch_stddev_layer_5322',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'group_size': 4,
    'num_new_features': 1,
    'eps': 1e-08,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    n, c, h, w = x.shape
    group_size = min(n, CONFIG['group_size'])
    num_new_features = CONFIG['num_new_features']
    y = jnp.reshape(x, (group_size, -1, num_new_features, c // num_new_features, h, w))
    y = y - jnp.mean(y, axis=0, keepdims=True)
    y = jnp.mean(y ** 2, axis=0)
    y = jnp.sqrt(y + CONFIG['eps'])
    y = jnp.mean(y, axis=(2, 3, 4), keepdims=True)
    y = jnp.mean(y, axis=2)
    y = jnp.tile(y, (group_size, 1, h, w))
    return jnp.concatenate([x, y], axis=1)
