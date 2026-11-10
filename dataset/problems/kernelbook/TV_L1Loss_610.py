"""TV-L1 loss: mean absolute vertical and horizontal finite differences, divided by batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TV_L1Loss_610',
    'batch': 16,
    'channels': 16,
    'height': 512,
    'width': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    b, c, h, w = x.shape
    count_h = c * (h - 1) * w
    count_w = c * h * (w - 1)
    h_tv = jnp.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    w_tv = jnp.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
    return (h_tv / count_h + w_tv / count_w) / b
