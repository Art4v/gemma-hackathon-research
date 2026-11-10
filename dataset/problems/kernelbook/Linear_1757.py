"""[bf16 variant] Linear_1757: matmul of a 4-D tensor over its last axis with a (in, out) weight, plus bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Linear_1757',
    'batch': 8,
    'height': 128,
    'width': 128,
    'in_channels': 256,
    'out_channels': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_src, k_weight, k_bias = jax.random.split(key, 3)
    src = jax.random.uniform(
        k_src,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['in_channels']),
        dtype,
    )
    weight = jax.random.normal(
        k_weight, (CONFIG['in_channels'], CONFIG['out_channels']), dtype
    ) * CONFIG['weight_scale']
    bias = jax.random.normal(k_bias, (CONFIG['out_channels'],), dtype) * CONFIG['weight_scale']
    return src, weight, bias


def workload(src, weight, bias):
    return jnp.matmul(src, weight) + bias
