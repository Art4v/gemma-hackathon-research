"""[bf16 variant] LayerNorm1D: (x - mean_c) / (unbiased_std_c + eps) * weight + bias, stats over axis 1, affine on last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm1D_9941',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'num_outputs': 128,  # must equal width: (1, num_outputs) broadcasts against the last axis
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jnp.ones((1, CONFIG['num_outputs']), dtype)
    bias = jnp.zeros((1, CONFIG['num_outputs']), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    mean = jnp.mean(x, axis=1, keepdims=True)
    std = jnp.std(x, axis=1, keepdims=True, ddof=1)  # torch.std is unbiased by default
    y = (x - mean) / (std + CONFIG['eps'])
    # torch expand_as aligns (1, N) with the trailing dims, so N scales the last axis
    return y * weight + bias
