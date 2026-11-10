"""InvHuberLoss: relu -> masked abs diff -> global max threshold -> berHu piecewise -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'InvHuberLoss_13590',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'ignore_index': 0.0,
    'c_scale': 0.2,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    kx, kt = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, shape, dtype)
    target = jax.random.uniform(kt, shape, dtype)
    return x, target


def workload(x, target):
    inp = jnp.maximum(x, 0)
    diff = inp - target
    mask = (target != CONFIG['ignore_index']).astype(diff.dtype)
    err = jnp.abs(diff * mask)
    c = CONFIG['c_scale'] * jnp.max(err)
    err2 = (diff ** 2 + c ** 2) / (2.0 * c)
    # Multiply by float masks like the PyTorch code (not jnp.where) so NaN/inf behaviour matches.
    mask_le = (err <= c).astype(diff.dtype)
    mask_gt = (err > c).astype(diff.dtype)
    return jnp.mean(err * mask_le + err2 * mask_gt)
