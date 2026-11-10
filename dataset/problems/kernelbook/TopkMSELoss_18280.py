"""[bf16 variant] TopkMSELoss: elementwise squared error -> mean(dim 2) -> mean(dim 1) -> top-k along last dim."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TopkMSELoss_18280',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'topk': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    output = jax.random.uniform(k1, shape, dtype)
    label = jax.random.uniform(k2, shape, dtype)
    return output, label


def workload(output, label):
    losses = jnp.square(output - label).mean(axis=2).mean(axis=1)
    values, _ = lax.top_k(losses, CONFIG['topk'])
    return values
