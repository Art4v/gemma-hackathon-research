"""[bf16 variant] BPR loss: embedding gather -> pairwise dot diff -> logsigmoid sum + L2 regularization."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BPR_12662',
    'user_size': 262144,
    'item_size': 262144,
    'dim': 128,
    'batch': 16384,
    'weight_decay': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    ku, ki, kj, kw, kh = jax.random.split(jax.random.key(0), 5)
    b = CONFIG['batch']
    u = jax.random.randint(ku, (b,), 0, CONFIG['user_size'], dtype=jnp.int32)
    i = jax.random.randint(ki, (b,), 0, CONFIG['item_size'], dtype=jnp.int32)
    j = jax.random.randint(kj, (b,), 0, CONFIG['item_size'], dtype=jnp.int32)
    W = jax.random.normal(kw, (CONFIG['user_size'], CONFIG['dim']), dtype) * 0.02
    H = jax.random.normal(kh, (CONFIG['item_size'], CONFIG['dim']), dtype) * 0.02
    return u, i, j, W, H


def workload(u, i, j, W, H):
    eu = jnp.take(W, u, axis=0)
    ei = jnp.take(H, i, axis=0)
    ej = jnp.take(H, j, axis=0)
    x_uij = jnp.sum(eu * ei, axis=1) - jnp.sum(eu * ej, axis=1)
    log_prob = jnp.sum(jax.nn.log_sigmoid(x_uij))
    reg = CONFIG['weight_decay'] * (jnp.sum(eu * eu) + jnp.sum(ei * ei) + jnp.sum(ej * ej))
    return -log_prob + reg
