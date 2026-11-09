"""SoftArgMax: full-length sum-pool of x and x*labels, ratio, argmax-gather over pooled axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftArgMax_3579',
    'rows': 8192,
    'length': 4096,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['rows'], CONFIG['length'])
    x = jax.random.uniform(k1, shape, dtype)
    labels = jax.random.uniform(k2, shape, dtype)
    return x, labels


def workload(x, labels):
    # kernel_size defaults to the full last dim, so avg_pool * k == sum with pooled length 1.
    y = jnp.sum(x * labels, axis=-1, keepdims=True)
    s = jnp.sum(x, axis=-1, keepdims=True)
    y = y / (s + CONFIG['eps'])
    ind = jnp.argmax(s, axis=-1)[..., None]
    return jnp.take_along_axis(y, ind, axis=-1)[..., 0]
