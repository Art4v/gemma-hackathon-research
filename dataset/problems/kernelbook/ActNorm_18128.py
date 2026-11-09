"""ActNorm data-dependent init forward: per-channel mean/var -> log_scale, shift -> affine (y, logdet)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ActNorm_18128',
    'batch': 8,
    'num_channels': 128,
    'height': 512,
    'width': 128,  # must equal num_channels: PyTorch's [None, :] broadcast aligns with the W axis
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    eps = CONFIG['eps']
    stat_axes = (0, 2, 3)
    mean = jnp.mean(x, axis=stat_axes)
    # ponytail: mirrors PyTorch's (1, C) broadcast against NCHW, which indexes by W, not C.
    zero_mean = x - mean[None, :]
    var = jnp.mean(zero_mean ** 2, axis=stat_axes)
    std = (var + eps) ** 0.5
    log_scale = jnp.log(1.0 / std)
    shift = -mean * jnp.exp(log_scale)
    logdet = jnp.sum(log_scale)
    y = x * jnp.exp(log_scale)[None, :] + shift[None, :]
    return y, logdet
