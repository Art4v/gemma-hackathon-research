"""[bf16 variant] SvmLoss: targets*2-1, hinge max(0, 1 - t*decisions), mean over all elements."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SvmLoss_3238',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    decisions = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    return decisions, targets


def workload(decisions, targets):
    t = targets.astype(jnp.float32) * 2 - 1
    projection_dist = 1 - t * decisions
    margin = jnp.maximum(projection_dist, 0)
    return jnp.mean(margin)
