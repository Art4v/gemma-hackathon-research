"""MSE loss (no reduction) -> masked fill of ignored targets -> optional per-sample smallest-k -> mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MSELossWithIgnore_6600',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'ignore_value': 4,
    'fraction': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_output, k_target = jax.random.split(key)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k_output, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return output, target


def workload(output, target):
    loss = jnp.square(output - target)
    loss = jnp.where(target == CONFIG['ignore_value'], jnp.zeros_like(loss), loss)
    if CONFIG['fraction'] < 1:
        loss = loss.reshape(loss.shape[0], -1)
        num_elements_to_keep = int(loss.shape[1] * CONFIG['fraction'])
        negated, _ = lax.top_k(-loss, num_elements_to_keep)
        loss = -negated
    return jnp.mean(loss)
