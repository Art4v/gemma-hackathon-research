"""[bf16 variant] ReLU followed by global L2 (Frobenius) norm: L2PenaltyConstraintLoss."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L2PenaltyConstraintLoss_11013',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)


def workload(x):
    gap_constraint = jax.nn.relu(x)
    return jnp.sqrt(jnp.sum(jnp.square(gap_constraint)))
