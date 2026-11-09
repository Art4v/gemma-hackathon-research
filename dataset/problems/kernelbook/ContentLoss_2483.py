"""ContentLoss: content_weight * sum((content_current - content_original) ** 2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContentLoss_2483',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_w, k_cur, k_orig = jax.random.split(jax.random.key(0), 3)
    content_weight = jax.random.uniform(k_w, shape, dtype)
    content_current = jax.random.uniform(k_cur, shape, dtype)
    content_original = jax.random.uniform(k_orig, shape, dtype)
    return content_weight, content_current, content_original


def workload(content_weight, content_current, content_original):
    # The weight is a full tensor in KernelBench's get_inputs, so the scalar sum broadcasts against it.
    diff = content_current - content_original
    return content_weight * jnp.sum(diff * diff)
