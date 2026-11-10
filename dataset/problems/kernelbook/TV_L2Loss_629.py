"""TV_L2Loss: NCHW forward differences along H and W -> squared sums -> count normalisation -> batch mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TV_L2Loss_629',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    batch_size, channels, h_x, w_x = x.shape
    count_h = channels * (h_x - 1) * w_x
    count_w = channels * h_x * (w_x - 1)
    h_tv = jnp.sum(jnp.square(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
    w_tv = jnp.sum(jnp.square(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
    return (h_tv / count_h + w_tv / count_w) / batch_size
