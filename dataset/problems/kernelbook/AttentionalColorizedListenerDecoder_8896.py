"""Rank-4 broadcast quadratic form -(c - mu) Sigma (c - mu) summed over axis 2, then softmax over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionalColorizedListenerDecoder_8896',
    'batch': 8,
    'contexts': 128,
    'color_dim': 128,
    'softmax_axis': 1,
    'sum_axis': 2,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['contexts'], CONFIG['color_dim'], CONFIG['color_dim'])
    k_color, k_mew, k_sigma = jax.random.split(jax.random.key(0), 3)
    color_seqs = jax.random.uniform(k_color, shape, dtype)
    mew = jax.random.uniform(k_mew, shape, dtype)
    sigma = jax.random.uniform(k_sigma, shape, dtype)
    return color_seqs, mew, sigma


def workload(color_seqs, mew, sigma):
    shifted = color_seqs - jnp.expand_dims(mew, 1)
    vec_mat = -jnp.matmul(jnp.expand_dims(shifted, 2), jnp.expand_dims(sigma, 1))
    # torch.squeeze(dim) is a no-op when the axis is not unit-sized.
    if vec_mat.shape[1] == 1:
        vec_mat = jnp.squeeze(vec_mat, axis=1)
    if vec_mat.shape[2] == 1:
        vec_mat = jnp.squeeze(vec_mat, axis=2)
    scores = jnp.sum(vec_mat * shifted, axis=CONFIG['sum_axis'])
    return jax.nn.softmax(scores, axis=CONFIG['softmax_axis'])
