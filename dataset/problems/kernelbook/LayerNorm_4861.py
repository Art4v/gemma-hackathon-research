"""[bf16 variant] LayerNorm_4861: per-sample mean/unbiased-std normalization over (C,H,W), divide by (std + eps), per-channel affine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_4861',
    'batch': 16,
    'num_features': 128,
    'height': 128,
    'width': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_features'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    gamma = jnp.ones((CONFIG['num_features'],), dtype)
    beta = jnp.zeros((CONFIG['num_features'],), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    axes = tuple(range(1, x.ndim))
    mean = jnp.mean(x, axis=axes, keepdims=True)
    # torch.std defaults to the unbiased estimator (correction=1).
    std = jnp.std(x, axis=axes, keepdims=True, ddof=1)
    y = (x - mean) / (std + CONFIG['eps'])
    affine_shape = (1, -1) + (1,) * (x.ndim - 2)
    return gamma.reshape(affine_shape) * y + beta.reshape(affine_shape)
