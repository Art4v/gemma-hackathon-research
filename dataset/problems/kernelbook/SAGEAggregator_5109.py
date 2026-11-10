"""[bf16 variant] SAGEAggregator: mean over neighbours (axis 1), linear projections of node and neighbour features, element-wise sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SAGEAggregator_5109',
    'batch': 8,
    'num_groups': 128,
    'num_nodes': 128,
    'num_neighbors': 8,
    'in_features': 128,
    'out_features': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_neigh, k_l, k_r = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['num_groups'], CONFIG['num_nodes'], CONFIG['in_features']),
        dtype,
    )
    neigh_x = jax.random.uniform(
        k_neigh,
        (CONFIG['num_groups'], CONFIG['num_neighbors'], CONFIG['num_nodes'], CONFIG['in_features']),
        dtype,
    )
    w_l = jax.random.normal(k_l, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    w_r = jax.random.normal(k_r, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    return x, neigh_x, w_l, w_r


def workload(x, neigh_x, w_l, w_r):
    neigh = jnp.mean(neigh_x, axis=1)
    return x @ w_l + neigh @ w_r
