"""ActNorm data-dependent init (axis-1 mean/var with last-axis broadcast) -> log_scale, shift -> affine transform + logdet."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ActNorm_1036',
    'batch': 16,
    'num_channels': 128,
    'height': 128,
    'width': 128,  # must equal num_channels: (1, C) parameters broadcast against the last axis
    'channel_axis': 1,
    'eps': 1e-05,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['num_channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    axes = tuple(i for i in range(x.ndim) if i != CONFIG['channel_axis'])
    mean = jnp.mean(x, axis=axes)
    zero_mean = x - mean[None, :]
    var = jnp.mean(zero_mean ** 2, axis=axes)
    std = (var + CONFIG['eps']) ** 0.5
    log_scale = jnp.log(1.0 / std)
    shift = -mean * jnp.exp(log_scale)
    logdet = jnp.sum(log_scale)
    return x * jnp.exp(log_scale[None, :]) + shift[None, :], logdet
