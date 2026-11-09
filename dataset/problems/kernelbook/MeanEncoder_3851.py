"""MeanEncoder: sum over the sequence axis, then divide by per-row length."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MeanEncoder_3851',
    'seq_len': 64,
    'batch': 512,
    'dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    kx, kl = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, (CONFIG['seq_len'], CONFIG['batch'], CONFIG['dim']), dtype)
    x_len = jax.random.randint(kl, (CONFIG['batch'],), 1, CONFIG['seq_len'] + 1, jnp.int32)
    return x, x_len


def workload(x, x_len):
    return jnp.sum(x, axis=0) / x_len.astype(jnp.float32).reshape(-1, 1)
