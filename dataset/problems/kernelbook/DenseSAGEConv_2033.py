"""DenseSAGEConv: adj @ x, divide by clamped degree, lin_rel(out) + lin_root(x) with root bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DenseSAGEConv_2033',
    'outer_batch': 8,
    'batch': 16,
    'num_nodes': 512,
    'in_channels': 512,
    'out_channels': 512,
    'degree_min': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_adj, k_rel, k_root, k_bias = jax.random.split(key, 5)
    a = CONFIG['outer_batch']
    b = CONFIG['batch']
    n = CONFIG['num_nodes']
    f_in = CONFIG['in_channels']
    f_out = CONFIG['out_channels']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, (a, b, n, f_in), dtype)
    adj = jax.random.uniform(k_adj, (b, n, n), dtype)
    w_rel = jax.random.normal(k_rel, (f_in, f_out), dtype) * scale
    w_root = jax.random.normal(k_root, (f_in, f_out), dtype) * scale
    bias = jax.random.normal(k_bias, (f_out,), dtype) * scale
    return x, adj, w_rel, w_root, bias


def workload(x, adj, w_rel, w_root, bias):
    out = jnp.matmul(adj, x)
    degree = jnp.sum(adj, axis=-1, keepdims=True)
    out = out / jnp.maximum(degree, CONFIG['degree_min'])
    return jnp.matmul(out, w_rel) + (jnp.matmul(x, w_root) + bias)
