"""InstanceNorm2d: per-(sample, channel) normalization over H*W (affine=False, eps=1e-5)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '34_InstanceNorm',
    'batch_size': 16,
    'features': 64,
    'dim1': 256,
    'dim2': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['features'], CONFIG['dim1'], CONFIG['dim2']),
        dtype,
    )
    return (x,)


def workload(x):
    eps = CONFIG['eps']
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=(2, 3), keepdims=True)
    centered = xf - mean
    var = jnp.mean(centered * centered, axis=(2, 3), keepdims=True)
    y = centered * jax.lax.rsqrt(var + eps)
    return y.astype(x.dtype)
