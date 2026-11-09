"""[bf16 variant] Focal Tversky loss: flatten -> three dot products -> Tversky index -> mean((1 - ti)^gamma)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalTiLoss_11047',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'alpha': 0.7,
    'beta': 0.4,
    'gamma': 0.75,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return output, target


def workload(output, target):
    pi = output.astype(jnp.float32).reshape(-1)
    gi = target.astype(jnp.float32).reshape(-1)
    # ponytail: sum of products instead of jnp.dot so TPU default precision stays f32
    intersection = jnp.sum(pi * gi)
    inter_alpha = jnp.sum((1.0 - pi) * gi)
    inter_beta = jnp.sum((1.0 - gi) * pi)
    eps = CONFIG['eps']
    ti = (intersection + eps) / (
        intersection + CONFIG['alpha'] * inter_alpha + CONFIG['beta'] * inter_beta + eps
    )
    return jnp.mean(jnp.power(1.0 - ti, CONFIG['gamma']))
