"""[bf16 variant] DiceCoeffLoss: flatten -> dot(predict, target) and sum(predict) + sum(target) -> (2*inter + eps) / (union + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DiceCoeffLoss_18224',
    'batch': 32,
    'channels': 64,
    'height': 128,
    'width': 128,
    'eps': 1e-4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    predict = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return predict, target


def workload(predict, target):
    eps = CONFIG['eps']
    p = predict.reshape(-1).astype(jnp.float32)
    t = target.reshape(-1).astype(jnp.float32)
    inter = jnp.dot(p, t)
    union = jnp.sum(p) + jnp.sum(t) + eps
    return (2.0 * inter + eps) / union
