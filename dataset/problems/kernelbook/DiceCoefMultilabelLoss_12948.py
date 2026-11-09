"""[bf16 variant] Softmax2d over channels -> per-label dice loss on the first num_labels channels -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceCoefMultilabelLoss_12948',
    'batch': 16,
    'channels': 4,
    'height': 512,
    'width': 512,
    'num_labels': 3,
    'smooth': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    predict = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return predict, target


def workload(predict, target):
    n = CONFIG['num_labels']
    smooth = CONFIG['smooth']
    p = jax.nn.softmax(predict, axis=1)[:, :n]
    t = target[:, :n].astype(p.dtype)
    axes = (0, 2, 3)
    inter = jnp.sum(p * t, axis=axes)
    score = (inter * 2.0 + smooth) / (jnp.sum(p, axis=axes) + jnp.sum(t, axis=axes) + smooth)
    return jnp.mean(1.0 - score)
