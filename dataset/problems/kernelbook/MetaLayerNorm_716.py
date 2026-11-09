"""MetaLayerNorm: layer_norm over the last dim with affine weight and bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MetaLayerNorm_716',
    'batch': 8,
    'd1': 64,
    'd2': 128,
    'normalized_shape': 1024,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['normalized_shape'])
    x = jax.random.uniform(key, shape, dtype)
    weight = jnp.ones((CONFIG['normalized_shape'],), dtype)
    bias = jnp.zeros((CONFIG['normalized_shape'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    # normalized_shape is 1D, so normalize over the trailing weight.ndim axes.
    axes = tuple(range(x.ndim - weight.ndim, x.ndim))
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=axes, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=axes, keepdims=True)
    y = (xf - mean) * jax.lax.rsqrt(var + CONFIG['eps'])
    y = y * weight.astype(jnp.float32) + bias.astype(jnp.float32)
    return y.astype(x.dtype)
