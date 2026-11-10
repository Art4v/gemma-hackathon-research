"""[bf16 variant] relu(weights), relu(beta), linear x @ w, then 1 - clamp(beta - x @ w, 0, 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'predicates1_8277',
    'batch': 8,
    'height': 128,
    'width': 128,
    'num_predicates': 256,
    'body_len': 256,
    'weight_scale': 0.02,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
    'complement': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['num_predicates']),
        dtype,
    )
    weights = jax.random.normal(
        k_w, (CONFIG['num_predicates'], CONFIG['body_len']), dtype
    ) * CONFIG['weight_scale']
    beta = jnp.ones((1, CONFIG['body_len']), dtype)
    return x, weights, beta


def workload(x, weights, beta):
    w = jax.nn.relu(weights)
    b = jax.nn.relu(beta)
    linear = jnp.matmul(x, w)
    clamped = jnp.clip(-linear + b, CONFIG['clamp_min'], CONFIG['clamp_max'])
    return CONFIG['complement'] - clamped
