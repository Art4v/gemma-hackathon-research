"""[bf16 variant] DenseSAGEConv: adj @ x, row-normalize by adj degree, @ W + b, L2 normalize."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DenseSAGEConv_1831',
    'batch': 8,
    'groups': 4,
    'num_nodes': 1024,
    'in_channels': 128,
    'out_channels': 128,
    'norm_eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_adj, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    b, g, n = CONFIG['batch'], CONFIG['groups'], CONFIG['num_nodes']
    c_in, c_out = CONFIG['in_channels'], CONFIG['out_channels']
    x = jax.random.uniform(k_x, (b, g, n, c_in), dtype)
    adj = jax.random.uniform(k_adj, (b, g, n, n), dtype)
    weight = jax.random.normal(k_w, (c_in, c_out), dtype) * 0.02
    bias = jax.random.normal(k_b, (c_out,), dtype) * 0.02
    return x, adj, weight, bias


def workload(x, adj, weight, bias):
    out = jnp.matmul(adj, x)
    out = out / jnp.sum(adj, axis=-1, keepdims=True)
    out = jnp.matmul(out, weight) + bias
    norm = jnp.sqrt(jnp.sum(out * out, axis=-1, keepdims=True))
    return out / jnp.maximum(norm, CONFIG['norm_eps'])
