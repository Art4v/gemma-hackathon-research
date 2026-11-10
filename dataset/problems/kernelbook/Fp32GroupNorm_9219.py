"""Fp32GroupNorm: float32 cast -> group mean/variance normalization -> per-channel scale and shift -> cast back."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Fp32GroupNorm_9219',
    'batch': 8,
    'num_channels': 128,
    'height': 256,
    'width': 256,
    'num_groups': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jnp.ones((CONFIG['num_channels'],), dtype)
    bias = jnp.zeros((CONFIG['num_channels'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    input_dtype = x.dtype
    shape = x.shape
    x32 = x.astype(jnp.float32)
    grouped = x32.reshape(shape[0], CONFIG['num_groups'], -1)
    mean = jnp.mean(grouped, axis=2, keepdims=True)
    var = jnp.var(grouped, axis=2, ddof=0, keepdims=True)
    normalized = (grouped - mean) / jnp.sqrt(var + CONFIG['eps'])
    normalized = normalized.reshape(shape)
    affine_shape = (1, shape[1]) + (1,) * (x32.ndim - 2)
    output = normalized * weight.astype(jnp.float32).reshape(affine_shape)
    output = output + bias.astype(jnp.float32).reshape(affine_shape)
    return output.astype(input_dtype)
