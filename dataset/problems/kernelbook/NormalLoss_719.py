"""[bf16 variant] NormalLoss: 1 - mean(dot(grad_fake, grad_real) / (||grad_fake|| * ||grad_real||)) along the last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NormalLoss_719',
    'batch': 16,
    'seq': 4096,
    'dim': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['seq'], CONFIG['dim'])
    grad_fake = jax.random.uniform(k1, shape, dtype)
    grad_real = jax.random.uniform(k2, shape, dtype)
    return grad_fake, grad_real


def workload(grad_fake, grad_real):
    prod = jnp.sum(grad_fake * grad_real, axis=-1)
    fake_norm = jnp.sqrt(jnp.sum(grad_fake ** 2, axis=-1))
    real_norm = jnp.sqrt(jnp.sum(grad_real ** 2, axis=-1))
    return 1 - jnp.mean(prod / (fake_norm * real_norm))
