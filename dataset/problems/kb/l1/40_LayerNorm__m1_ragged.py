"""[m1 M3_ragged variant of kb/l1/40_LayerNorm] LayerNorm over the trailing (features, dim1, dim2) axes with affine scale/shift, with dim2 at 200 instead of 256 so the last axis is not a multiple of 128."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '40_LayerNorm_m1_ragged',
    'batch_size': 16,
    'features': 64,
    'dim1': 256,
    'dim2': 200,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['features'], CONFIG['dim1'], CONFIG['dim2'])
    x = jax.random.uniform(key, shape, dtype)
    norm_shape = (CONFIG['features'], CONFIG['dim1'], CONFIG['dim2'])
    weight = jnp.ones(norm_shape, dtype)
    bias = jnp.zeros(norm_shape, dtype)
    return (x, weight, bias)


def workload(x, weight, bias):
    eps = CONFIG['eps']
    n_axes = weight.ndim
    axes = tuple(range(x.ndim - n_axes, x.ndim))
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=axes, keepdims=True)
    centered = xf - mean
    var = jnp.mean(centered * centered, axis=axes, keepdims=True)
    y = centered * jax.lax.rsqrt(var + eps)
    y = y * weight.astype(jnp.float32) + bias.astype(jnp.float32)
    return y.astype(x.dtype)
