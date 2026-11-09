"""[bf16 variant] AutoregressiveShift: concat(broadcast(first_token), x[:-1]) along the sequence axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AutoregressiveShift_902',
    'seq_len': 1024,
    'batch': 32,
    'embed_dim': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_t = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_x, (CONFIG['seq_len'], CONFIG['batch'], CONFIG['embed_dim']), dtype)
    first_token = jax.random.normal(k_t, (1, 1, CONFIG['embed_dim']), dtype) * 0.02
    return x, first_token


def workload(x, first_token):
    first = jnp.broadcast_to(first_token, (1, x.shape[1], x.shape[2]))
    return jnp.concatenate([first, x[:-1]], axis=0)
