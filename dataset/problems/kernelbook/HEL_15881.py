"""[bf16 variant] HEL loss: 5x5 avg-pool edge mask, edge-weighted L1 plus foreground/background region loss, mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'HEL_15881',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 5,
    'stride': 1,
    'padding': 2,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_target = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def _avg_pool2d(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    nhwc = jnp.transpose(x, (0, 2, 3, 1))
    summed = lax.reduce_window(
        nhwc,
        0.0,
        lax.add,
        window_dimensions=(1, k, k, 1),
        window_strides=(1, s, s, 1),
        padding=((0, 0), (p, p), (p, p), (0, 0)),
    )
    pooled = summed / (k * k)
    return jnp.transpose(pooled, (0, 3, 1, 2))


def _edge_loss(pred, target):
    eps = CONFIG['eps']
    edge = target - _avg_pool2d(target)
    edge = (edge != 0).astype(target.dtype)
    numerator = (edge * jnp.abs(pred - target)).sum(axis=(2, 3))
    denominator = edge.sum(axis=(2, 3)) + eps
    return numerator / denominator


def _region_loss(pred, target):
    eps = CONFIG['eps']
    numerator_fore = (target - target * pred).sum(axis=(2, 3))
    denominator_fore = target.sum(axis=(2, 3)) + eps
    numerator_back = ((1 - target) * pred).sum(axis=(2, 3))
    denominator_back = (1 - target).sum(axis=(2, 3)) + eps
    return numerator_fore / denominator_fore + numerator_back / denominator_back


def workload(pred, target):
    return (_edge_loss(pred, target) + _region_loss(pred, target)).mean()
