"""Log-softmax over channels followed by KL-divergence loss (mean reduction) against soft targets."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftTargetCrossEntropy_17728',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_t = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k_x, shape, dtype)
    target = jax.random.uniform(k_t, shape, dtype)
    return x, target


def workload(x, target):
    logp = jax.nn.log_softmax(x, axis=CONFIG['softmax_axis'])
    # PyTorch kl_div: xlogy(t, t) - t * input, so t == 0 contributes 0.
    pointwise = jax.scipy.special.xlogy(target, target) - target * logp
    return jnp.mean(pointwise)
