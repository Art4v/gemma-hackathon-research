"""Channel-axis softmax followed by mean-squared-error against a target (MSELoss2d)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MSELoss2d_12703',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_output, k_target = jax.random.split(key)
    output = jax.random.uniform(k_output, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return output, target


def workload(output, target):
    probs = jax.nn.softmax(output, axis=CONFIG['softmax_axis'])
    return jnp.mean(jnp.square(probs - target))
