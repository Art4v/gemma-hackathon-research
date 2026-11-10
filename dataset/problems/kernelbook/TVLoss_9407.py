"""TVLoss_9407: squared forward differences along H and W -> per-axis sums -> count-normalized sum -> weighted batch-mean scalar."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TVLoss_9407',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'tv_loss_weight': 1,
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
    batch_size, channels, h_x, w_x = x.shape
    count_h = channels * (h_x - 1) * w_x
    count_w = channels * h_x * (w_x - 1)
    h_tv = jnp.sum(jnp.square(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]))
    w_tv = jnp.sum(jnp.square(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]))
    return CONFIG['tv_loss_weight'] * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
