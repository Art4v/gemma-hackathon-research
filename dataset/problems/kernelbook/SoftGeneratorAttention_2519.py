"""Elementwise mul -> sum over channels -> softmax over batch axis (PyTorch implicit dim=0 for 3D)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftGeneratorAttention_2519',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'reduce_dim': 1,
    # F.softmax without dim on a 3D tensor resolves to dim=0 in PyTorch.
    'softmax_dim': 0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    key = jax.random.uniform(k1, shape, dtype)
    x = jax.random.uniform(k2, shape, dtype)
    return key, x


def workload(key, x):
    attn = jnp.sum(key * x, axis=CONFIG['reduce_dim'])
    return jax.nn.softmax(attn, axis=CONFIG['softmax_dim'])
