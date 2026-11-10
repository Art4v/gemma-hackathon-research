"""[bf16 variant] MatchingTensor: L2-normalize x and y, then bilinear einsum bld,cde,bre->bclr with a channel interaction tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MatchingTensor_4700',
    'batch': 8,
    'left_len': 256,
    'right_len': 256,
    'matching_dim': 512,
    'channels': 128,
    'eps': 1e-12,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky, kw = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['left_len'], CONFIG['matching_dim']), dtype)
    y = jax.random.uniform(
        ky, (CONFIG['batch'], CONFIG['right_len'], CONFIG['matching_dim']), dtype)
    w = jax.random.normal(
        kw, (CONFIG['channels'], CONFIG['matching_dim'], CONFIG['matching_dim']), dtype
    ) * CONFIG['weight_scale']
    return x, y, w


def workload(x, y, w):
    eps = CONFIG['eps']
    x = x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)
    y = y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), eps)
    xw = jnp.einsum('bld,cde->bcle', x, w)
    return jnp.einsum('bcle,bre->bclr', xw, y)
