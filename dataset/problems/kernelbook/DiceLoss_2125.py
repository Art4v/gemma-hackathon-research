"""DiceLoss: softmax(dim=1) -> global sum(p*t)+eps / sum(p+t)+eps -> 1 - dice."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceLoss_2125',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'epsilon': 1e-05,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return output, target


def workload(output, target):
    eps = CONFIG['epsilon']
    p = jax.nn.softmax(output, axis=1)
    # Flatten to (C, -1) then sum(-1).sum() in torch is just a full reduction.
    intersect = jnp.sum(p * target) + eps
    denominator = jnp.sum(p + target) + eps
    return 1.0 - intersect / denominator
