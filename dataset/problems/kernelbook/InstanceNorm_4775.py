"""InstanceNorm: per-(batch, channel) normalization over spatial dims via E[x^2]-E[x]^2, then affine scale/shift."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'InstanceNorm_4775',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    scale = jnp.ones((CONFIG['channels'],), dtype)
    shift = jnp.zeros((CONFIG['channels'],), dtype)
    return x, scale, shift


def workload(x, scale, shift):
    b, c = x.shape[0], x.shape[1]
    xf = x.reshape(b, c, -1)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    mean_x2 = jnp.mean(xf * xf, axis=-1, keepdims=True)
    var = mean_x2 - mean * mean
    x_norm = (xf - mean) / jnp.sqrt(var + CONFIG['eps'])
    out = scale.reshape(1, -1, 1) * x_norm + shift.reshape(1, -1, 1)
    return out.reshape(x.shape)
