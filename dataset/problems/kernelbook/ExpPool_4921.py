"""[bf16 variant] ExpPool: spatial max-shift, exp, normalize by (sum + eps), weighted spatial sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ExpPool_4921',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'epsilon': 1e-07,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    feat_map = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (feat_map,)


def workload(feat_map):
    m = jnp.max(feat_map, axis=(-2, -1), keepdims=True)
    e = jnp.exp(feat_map - m)
    sum_exp = jnp.sum(e, axis=(-2, -1), keepdims=True) + CONFIG['epsilon']
    exp_weight = e / sum_exp
    return jnp.sum(feat_map * exp_weight, axis=(-2, -1), keepdims=True)
