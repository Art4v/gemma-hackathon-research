"""[bf16 variant] MaskedLoss: diff = (y - Y) / 5; mean(square(diff[idx])) with integer-index gather along dim 0."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaskedLoss_12037',
    'shape': (32, 64, 128, 128),
    'num_indices': 32,
    'divisor': 5.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    shape = CONFIG['shape']
    y = jax.random.uniform(k1, shape, dtype)
    Y = jax.random.uniform(k2, shape, dtype)
    # torch passes an int64 tensor, so diff[mask] is integer indexing along dim 0, not boolean masking.
    mask = jax.random.randint(k3, (CONFIG['num_indices'],), 0, shape[0], dtype=jnp.int32)
    return y, Y, mask


def workload(y, Y, mask):
    diff = (y - Y) / CONFIG['divisor']
    return jnp.mean(jnp.square(jnp.take(diff, mask, axis=0)))
