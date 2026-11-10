"""[bf16 variant] WDLoss: mean(t_x), mean(t_y), axis-1 L2 norm of t_z, squared-deviation penalty, negated Wasserstein-GP loss."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WDLoss_4016',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'norm_axis': 1,
    'target_norm': 1.0,
    'lambda': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_x, k_y, k_z = jax.random.split(key, 3)
    t_x = jax.random.uniform(k_x, shape, dtype)
    t_y = jax.random.uniform(k_y, shape, dtype)
    t_z = jax.random.uniform(k_z, shape, dtype)
    return t_x, t_y, t_z


def workload(t_x, t_y, t_z):
    norms = jnp.sqrt(jnp.sum(jnp.square(t_z), axis=CONFIG['norm_axis']))
    penalty = jnp.mean(jnp.square(norms - CONFIG['target_norm']))
    return -(jnp.mean(t_x) - jnp.mean(t_y) - CONFIG['lambda'] * penalty)
