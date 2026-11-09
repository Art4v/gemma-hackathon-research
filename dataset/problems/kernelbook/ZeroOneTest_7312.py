"""[bf16 variant] Zero-one test loss: prior * mean((1 - sign(p)) / 2) + (1 - prior) * mean((1 + sign(n)) / 2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ZeroOneTest_7312',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_p, k_n, k_prior = jax.random.split(jax.random.key(0), 3)
    output_p = jax.random.uniform(k_p, shape, dtype)
    output_n = jax.random.uniform(k_n, shape, dtype)
    prior = jax.random.uniform(k_prior, shape, dtype)
    return output_p, output_n, prior


def workload(output_p, output_n, prior):
    pos_err = jnp.mean((1 - jnp.sign(output_p)) / 2)
    neg_err = jnp.mean((1 + jnp.sign(output_n)) / 2)
    return prior * pos_err + (1 - prior) * neg_err
