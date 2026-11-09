"""[bf16 variant] Sparsemax over the last axis: sort desc + cumsum + support threshold + clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sparsemax',
    'batch': 8,
    'seq_len': 512,
    'dim': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.normal(key, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim']), dtype)
    return (x,)


def workload(x):
    n = x.shape[-1]
    z_sorted = jnp.flip(jnp.sort(x, axis=-1), axis=-1)
    cumsum = jnp.cumsum(z_sorted, axis=-1)
    r = jnp.arange(1, n + 1, dtype=x.dtype)
    support = (1 + r * z_sorted) > cumsum
    k = jnp.maximum(jnp.sum(support, axis=-1, keepdims=True), 1).astype(x.dtype)
    support_sum = jnp.sum(z_sorted * support.astype(x.dtype), axis=-1, keepdims=True)
    tau = (support_sum - 1) / k
    return jnp.maximum(x - tau, 0)
