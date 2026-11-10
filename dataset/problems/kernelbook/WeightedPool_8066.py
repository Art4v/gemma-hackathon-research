"""[bf16 variant] WeightedPool: tensordot with weight, mask_logits broadcast, softmax over axis 1, matmul with transposed x, conditional squeeze."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WeightedPool_8066',
    'batch': 8,
    'dim': 128,
    'mask_outer': 8,
    'mask_last': 128,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_mask, k_weight = jax.random.split(key, 3)
    batch = CONFIG['batch']
    dim = CONFIG['dim']
    x = jax.random.uniform(k_x, (batch, dim, dim, dim), dtype)
    mask = jax.random.uniform(k_mask, (CONFIG['mask_outer'], batch, dim, CONFIG['mask_last']), dtype)
    weight = jax.random.normal(k_weight, (dim, 1), dtype) * 0.02
    return x, mask, weight


def workload(x, mask, weight):
    alpha = jnp.tensordot(x, weight, axes=1)
    mask = jnp.expand_dims(mask.astype(jnp.float32), 2)
    alpha = alpha + (1.0 - mask) * CONFIG['mask_value']
    alphas = jax.nn.softmax(alpha, axis=1)
    pooled = jnp.matmul(jnp.swapaxes(x, 1, 2), alphas)
    if pooled.shape[2] == 1:
        pooled = jnp.squeeze(pooled, axis=2)
    return pooled
