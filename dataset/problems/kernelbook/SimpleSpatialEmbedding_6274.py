"""[bf16 variant] SimpleSpatialEmbedding: matmul(x, b) followed by concat([sin, cos], axis=-1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimpleSpatialEmbedding_6274',
    'batch': 8,
    'height': 128,
    'width': 128,
    'in_features': 512,
    'out_features': 512,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_b = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['in_features']),
        dtype,
    )
    b = jax.random.normal(
        k_b,
        (CONFIG['in_features'], CONFIG['out_features']),
        dtype,
    ) * CONFIG['weight_scale']
    return x, b


def workload(x, b):
    y = jnp.matmul(x, b)
    return jnp.concatenate([jnp.sin(y), jnp.cos(y)], axis=-1)
