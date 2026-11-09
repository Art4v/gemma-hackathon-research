"""[bf16 variant] SoftDiceLossSquared: x*y and x^2+y^2 summed over spatial axes, then -mean(2*(I+s)/(D+s))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftDiceLossSquared_13375',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'smooth': 1.0,
    'batch_dice': False,
    'do_bg': True,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    # Shapes match, so y is used directly as y_onehot (no scatter branch).
    spatial = tuple(range(2, x.ndim))
    axes = (0,) + spatial if CONFIG['batch_dice'] else spatial
    smooth = CONFIG['smooth']
    intersect = jnp.sum(x * y, axis=axes) + smooth
    denominator = jnp.sum(x * x + y * y, axis=axes) + smooth
    dc = 2 * intersect / denominator
    if not CONFIG['do_bg']:
        dc = dc[1:] if CONFIG['batch_dice'] else dc[:, 1:]
    return -jnp.mean(dc)
