"""CnptAttention: negative squared L2 distance to two key sets, max, softmax over last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CnptAttention_18225',
    'num': 512,
    'dim': 128,
    'num_key_sets': 2,
}


def create_inputs(dtype=jnp.float32):
    k_q, k_k = jax.random.split(jax.random.key(0))
    n, d = CONFIG['num'], CONFIG['dim']
    query = jax.random.uniform(k_q, (1, d), dtype)
    key = jax.random.uniform(k_k, (CONFIG['num_key_sets'], n, n, d), dtype)
    return query, key


def workload(query, key):
    # query.view(1, D).expand(num, -1) gives identical rows, so broadcasting the single row is equivalent.
    q = query.reshape(query.shape[-1])
    h_score = -jnp.sum((q - key[0]) ** 2, axis=-1)
    t_score = -jnp.sum((q - key[1]) ** 2, axis=-1)
    score = jnp.maximum(h_score, t_score)
    return jnp.squeeze(jax.nn.softmax(score, axis=-1))
