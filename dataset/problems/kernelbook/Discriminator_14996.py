"""Discriminator_14996: expand S to node shape -> bilinear(node, S) + bias -> conditional squeeze(dim 1) -> sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Discriminator_14996',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'n_in': 128,
    'n_out': 128,
    'out_features': 1,
    'squeeze_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_s, k_node, k_weight, k_bias = jax.random.split(key, 4)
    S = jax.random.uniform(
        k_s, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_out']), dtype
    )
    node = jax.random.uniform(
        k_node, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_in']), dtype
    )
    weight = jax.random.normal(
        k_weight, (CONFIG['out_features'], CONFIG['n_in'], CONFIG['n_out']), dtype
    ) * CONFIG['weight_scale']
    bias = jax.random.normal(k_bias, (CONFIG['out_features'],), dtype) * CONFIG['weight_scale']
    return S, node, weight, bias


def workload(S, node, weight, bias):
    S = jnp.broadcast_to(S, node.shape)
    left = jnp.matmul(node[..., None, :], weight)
    score = jnp.sum(left * S[..., None, :], axis=-1) + bias
    axis = CONFIG['squeeze_dim']
    if score.shape[axis] == 1:
        score = jnp.squeeze(score, axis=axis)
    return jax.nn.sigmoid(score)
