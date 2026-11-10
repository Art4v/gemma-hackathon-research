"""ShrinkageLoss_6501: channel-axis L2 norm of (estimate - ground_truth) -> squared norm / (1 + exp(speed * (loc - norm))) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ShrinkageLoss_6501',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'norm_axis': 1,
    'shrink_speed': 10.0,
    'shrink_loc': 0.2,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_estimate, k_ground_truth = jax.random.split(key)
    estimate = jax.random.uniform(k_estimate, shape, dtype)
    ground_truth = jax.random.uniform(k_ground_truth, shape, dtype)
    return estimate, ground_truth


def workload(estimate, ground_truth):
    diff = estimate - ground_truth
    l2_loss = jnp.sqrt(jnp.sum(jnp.square(diff), axis=CONFIG['norm_axis']))
    shrink_loss = (l2_loss * l2_loss) / (
        1.0 + jnp.exp(CONFIG['shrink_speed'] * (CONFIG['shrink_loc'] - l2_loss))
    )
    return jnp.mean(shrink_loss)
