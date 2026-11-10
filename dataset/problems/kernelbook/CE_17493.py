"""[bf16 variant] Binary cross-entropy chain: log(p + eps), log(1 - p + eps), weighted sum, negated mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CE_17493',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    mat1 = jax.random.uniform(k1, shape, dtype)
    mat2 = jax.random.uniform(k2, shape, dtype)
    return mat1, mat2


def workload(mat1, mat2):
    eps = CONFIG['eps']
    return -jnp.mean(mat2 * jnp.log(mat1 + eps) + (1 - mat2) * jnp.log(1 - mat1 + eps))
