"""[bf16 variant] FBetaLoss: sigmoid -> channel-sum precision/recall -> F-beta -> 1 - mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FBetaLoss_10729',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'beta': 1.0,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    inputs = jax.random.uniform(k1, shape, dtype)
    target = jax.random.uniform(k2, shape, dtype)
    return inputs, target


def workload(inputs, target):
    eps = CONFIG['eps']
    beta2 = CONFIG['beta'] ** 2
    p = jax.nn.sigmoid(inputs)
    tp = jnp.sum(p * target, axis=1)
    precision = tp / (jnp.sum(p, axis=1) + eps)
    recall = tp / (jnp.sum(target, axis=1) + eps)
    fbeta = jnp.mean((precision * recall) / (precision * beta2 + recall + eps) * (1 + beta2))
    return 1 - fbeta
