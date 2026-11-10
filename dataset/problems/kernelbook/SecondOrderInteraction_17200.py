"""[bf16 variant] Factorization-machine second-order interaction: square, two matmuls (x @ v, x^2 @ v^2), square, subtract, sum over factors, scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SecondOrderInteraction_17200',
    'batch': 4096,
    'n_features': 4096,
    'n_factors': 1024,
    'scale': 0.5,
    'weight_std': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_v = jax.random.split(key)
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['n_features']), dtype)
    v = jax.random.normal(k_v, (CONFIG['n_features'], CONFIG['n_factors']), dtype) * CONFIG['weight_std']
    return x, v


def workload(x, v):
    pow_x = jnp.square(x)
    pow_v = jnp.square(v)
    pow_sum = jnp.square(x @ v)
    sum_pow = pow_x @ pow_v
    return CONFIG['scale'] * jnp.sum(pow_sum - sum_pow, axis=1, keepdims=True)
