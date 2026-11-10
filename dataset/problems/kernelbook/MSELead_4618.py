"""[bf16 variant] Per-channel MSE loss (over dim 1) followed by mean across channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MSELead_4618',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    t = jax.random.uniform(k2, shape, dtype)
    return x, t


def workload(x, t):
    sq = jnp.square(x - t)
    reduce_axes = tuple(a for a in range(x.ndim) if a != 1)
    per_channel = jnp.mean(sq, axis=reduce_axes)
    return jnp.mean(per_channel)
