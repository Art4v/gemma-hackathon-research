"""SmoothL1 loss (beta=1) -> masked_fill(target == ignore_value, 0) -> optional per-row bottom-k -> mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SmoothL1LossWithIgnore_6607',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 256,
    'ignore_value': 4,
    'fraction': 1.0,
    'beta': 1.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_out, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return output, target


def workload(output, target):
    beta = CONFIG['beta']
    diff = jnp.abs(output - target)
    loss = jnp.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    loss = jnp.where(target == CONFIG['ignore_value'], jnp.zeros_like(loss), loss)
    if CONFIG['fraction'] < 1:
        loss = loss.reshape(loss.shape[0], -1)
        k = int(loss.shape[1] * CONFIG['fraction'])
        neg_vals, _ = lax.top_k(-loss, k)
        loss = -neg_vals
    return jnp.mean(loss)
