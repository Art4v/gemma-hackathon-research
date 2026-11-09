"""[bf16 variant] SoftCrossEntropy: -log_softmax(x, axis=1) * target (broadcast over trailing H,W), summed, divided by target.shape[0]."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftCrossEntropy_16792',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k1, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    # Target broadcasts against the trailing (H, W) dims, as in the PyTorch original.
    target = jax.random.uniform(k2, (CONFIG['height'], CONFIG['width']), dtype)
    return x, target


def workload(x, target):
    log_likelihood = -jax.nn.log_softmax(x, axis=CONFIG['softmax_axis'])
    sample_num = target.shape[0]
    return jnp.sum(log_likelihood * target) / sample_num
