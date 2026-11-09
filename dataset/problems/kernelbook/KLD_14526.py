"""KL divergence: softmax(targets, dim=1) vs log_softmax(inputs, dim=1), kl_div batchmean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KLD_14526',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    targets = jax.random.uniform(k1, shape, dtype)
    inputs = jax.random.uniform(k2, shape, dtype)
    return targets, inputs


def workload(targets, inputs):
    axis = CONFIG['softmax_axis']
    # log of softmax(targets) via log_softmax: same value as xlogy(t, t) / t, more stable.
    log_t = jax.nn.log_softmax(targets, axis=axis)
    t = jnp.exp(log_t)
    log_p = jax.nn.log_softmax(inputs, axis=axis)
    return jnp.sum(t * (log_t - log_p)) / targets.shape[0]
