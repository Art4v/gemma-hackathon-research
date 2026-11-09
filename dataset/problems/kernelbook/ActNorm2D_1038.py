"""ActNorm2D data-dependent init forward: per-channel batch mean/std normalize plus logdet."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ActNorm2D_1038',
    'batch': 16,
    'num_channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    # First forward call: parameters are overwritten from batch statistics before use.
    mean = jnp.mean(x, axis=(0, 2, 3))
    zero_mean = x - mean[None, :, None, None]
    var = jnp.mean(zero_mean ** 2, axis=(0, 2, 3))
    std = (var + CONFIG['eps']) ** 0.5
    log_scale = jnp.log(1.0 / std)
    shift = -mean * jnp.exp(log_scale)
    logdet = jnp.sum(log_scale) * x.shape[2] * x.shape[3]
    out = x * jnp.exp(log_scale)[None, :, None, None] + shift[None, :, None, None]
    return out, logdet
