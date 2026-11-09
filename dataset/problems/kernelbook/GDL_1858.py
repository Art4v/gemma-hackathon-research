"""GDL: generalized Dice loss (tp/fp/fn spatial sums, volume weighting, channel sum, dice mean)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GDL_1858',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
    'smooth': 1.0,
    'volume_eps': 1e-6,
    'loss_weight': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    y = jax.random.uniform(k2, shape, dtype)
    return x, y


def workload(x, y):
    # y has the same shape as x, so PyTorch uses it directly as y_onehot (no scatter).
    axes = tuple(range(2, x.ndim))
    tp = jnp.sum(x * y, axis=axes)
    fp = jnp.sum(x * (1 - y), axis=axes)
    fn = jnp.sum((1 - x) * y, axis=axes)
    volumes = jnp.sum(y, axis=axes) + CONFIG['volume_eps']
    tp = jnp.sum(tp / volumes, axis=1)
    fp = jnp.sum(fp / volumes, axis=1)
    fn = jnp.sum(fn / volumes, axis=1)
    smooth = CONFIG['smooth']
    dc = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    return -jnp.mean(dc) * CONFIG['loss_weight']
