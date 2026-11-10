"""[bf16 variant] Categorical KL divergence: exp(log_qy) * (log_qy - log_py) summed over the latent axis, no reduction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CatKLLoss_10289',
    'batch': 8,
    'latent_size': 128,
    'height': 128,
    'width': 128,
    'sum_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['latent_size'], CONFIG['height'], CONFIG['width'])
    k_q, k_p = jax.random.split(jax.random.key(0))
    log_qy = jax.random.uniform(k_q, shape, dtype)
    log_py = jax.random.uniform(k_p, shape, dtype)
    return log_qy, log_py


def workload(log_qy, log_py):
    qy = jnp.exp(log_qy)
    return jnp.sum(qy * (log_qy - log_py), axis=CONFIG['sum_axis'])
