"""Binary dice loss: 1 - (2*sum(p*t) + smooth) / (sum(p^P + t^P) + smooth)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BinaryDiceLoss_6463',
    'batch': 16,
    'channels': 8,
    'height': 512,
    'width': 512,
    'smooth': 1.0,
    'p': 2,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    predict = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return predict, target


def workload(predict, target):
    smooth = CONFIG['smooth']
    p = CONFIG['p']
    # Sums run over the whole tensor, batch included, as in the PyTorch module.
    num = jnp.sum(predict * target) * 2 + smooth
    den = jnp.sum(predict ** p + target ** p) + smooth
    return 1 - num / den
