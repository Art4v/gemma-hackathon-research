"""GroupScaling1D: per-group mean of x^2, repeat over group, x / sqrt(moment2 + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GroupScaling1D_73',
    'T': 512,
    'B': 64,
    'C': 1024,
    'group_num': 4,
    'eps': 1e-05,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['T'], CONFIG['B'], CONFIG['C']), dtype)
    return (x,)


def workload(x):
    T, B, C = x.shape[0], x.shape[1], x.shape[2]
    g = CONFIG['group_num']
    cg = C // g
    xg = x.reshape(T, B, g, cg)
    m2 = jnp.mean(xg * xg, axis=3, keepdims=True)
    m2 = jnp.broadcast_to(m2, (T, B, g, cg)).reshape(T, B, C)
    # Divide the raw input by the (T, B, C) moment so broadcasting matches torch for any input rank.
    return x / jnp.sqrt(m2 + CONFIG['eps'])
