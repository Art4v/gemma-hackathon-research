"""[bf16 variant] HyperSphereLoss: project to hypersphere (concat), squared L2 norm over dim 1, acos of north-pole distance, mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HyperSphereLoss_7995',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def _project_to_hypersphere(v):
    v_norm = jnp.sum(v * v, axis=1, keepdims=True)
    a = 2 * v / (v_norm + 1)
    b = (v_norm - 1) / (v_norm + 1)
    return jnp.concatenate([a, b], axis=1)


def workload(x):
    q = _project_to_hypersphere(x)
    q_norm = jnp.sum(q * q, axis=1)
    loss = 2 * q[:, -1] / (1 + q_norm)
    return jnp.mean(jnp.arccos(loss))
