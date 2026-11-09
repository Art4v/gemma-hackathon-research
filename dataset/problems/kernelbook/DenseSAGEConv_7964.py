"""DenseSAGEConv: (adj + I) @ x, divide by row sums, @ W + b, L2-normalize the last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DenseSAGEConv_7964',
    'adj_outer': 4,
    'batch': 8,
    'num_nodes': 1024,
    'in_channels': 512,
    'out_channels': 512,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_adj, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    a, b, n = CONFIG['adj_outer'], CONFIG['batch'], CONFIG['num_nodes']
    f_in, f_out = CONFIG['in_channels'], CONFIG['out_channels']
    x = jax.random.uniform(k_x, (b, n, f_in), dtype)
    adj = jax.random.uniform(k_adj, (a, b, n, n), dtype)
    weight = jax.random.normal(k_w, (f_in, f_out), dtype) * 0.02
    bias = jax.random.normal(k_b, (f_out,), dtype) * 0.02
    return x, adj, weight, bias


def workload(x, adj, weight, bias):
    n = x.shape[-2]
    adj = adj + jnp.eye(n, dtype=adj.dtype)
    # adj (A, B, N, N) @ x (B, N, F) broadcasts to (A, B, N, F), same as torch.matmul.
    out = jnp.matmul(adj, x)
    out = out / jnp.sum(adj, axis=-1, keepdims=True)
    out = jnp.matmul(out, weight) + bias
    norm = jnp.sqrt(jnp.sum(out * out, axis=-1, keepdims=True))
    return out / jnp.maximum(norm, CONFIG['eps'])
