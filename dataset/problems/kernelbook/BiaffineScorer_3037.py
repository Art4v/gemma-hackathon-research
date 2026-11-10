"""Append constant-1 feature to both inputs, then bilinear form x1^T W_k x2 + b_k (BiaffineScorer)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BiaffineScorer_3037',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input1_size': 127,
    'input2_size': 127,
    'output_size': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    prefix = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'])
    input1 = jax.random.uniform(k1, prefix + (CONFIG['input1_size'],), dtype)
    input2 = jax.random.uniform(k2, prefix + (CONFIG['input2_size'],), dtype)
    weight_shape = (
        CONFIG['output_size'],
        CONFIG['input1_size'] + 1,
        CONFIG['input2_size'] + 1,
    )
    weight = jax.random.normal(k3, weight_shape, dtype) * CONFIG['weight_scale']
    bias = jax.random.normal(k4, (CONFIG['output_size'],), dtype) * CONFIG['weight_scale']
    return input1, input2, weight, bias


def workload(input1, input2, weight, bias):
    x1 = jnp.concatenate(
        [input1, jnp.ones(input1.shape[:-1] + (1,), input1.dtype)], axis=-1
    )
    x2 = jnp.concatenate(
        [input2, jnp.ones(input2.shape[:-1] + (1,), input2.dtype)], axis=-1
    )
    bilinear = lax.map(lambda w: jnp.sum((x1 @ w) * x2, axis=-1), weight)
    return jnp.moveaxis(bilinear, 0, -1) + bias
