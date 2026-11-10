"""kAttentionPooling: matmul(x, theta_k) -> softmax over sequence axis -> einsum('nij,nik->nkj') pooling."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'kAttentionPooling_7782',
    'batch': 64,
    'seq_len': 512,
    'hidden_size': 1024,
    'k_heads': 128,
    'softmax_axis': -2,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_theta = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_size']),
        dtype,
    )
    theta_k = jax.random.normal(
        k_theta,
        (CONFIG['hidden_size'], CONFIG['k_heads']),
        dtype,
    ) * CONFIG['weight_scale']
    return x, theta_k


def workload(x, theta_k):
    attention = jnp.matmul(x, theta_k)
    attention = jax.nn.softmax(attention, axis=CONFIG['softmax_axis'])
    return jnp.einsum('nij,nik->nkj', x, attention)
