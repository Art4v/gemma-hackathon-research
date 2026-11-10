"""DenseSAGEConv: add self-loops -> adj @ x -> degree normalize (clamp min 1) -> x @ weight + bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DenseSAGEConv_1760',
    'extra_dim': 8,
    'batch': 8,
    'num_nodes': 512,
    'in_channels': 512,
    'out_channels': 512,
    'loop_value': 1.0,
    'min_degree': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_adj, k_w, k_b = jax.random.split(key, 4)
    E = CONFIG['extra_dim']
    B = CONFIG['batch']
    N = CONFIG['num_nodes']
    F = CONFIG['in_channels']
    O = CONFIG['out_channels']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, (E, B, N, F), dtype)
    adj = jax.random.uniform(k_adj, (B, N, N), dtype)
    weight = jax.random.normal(k_w, (F, O), dtype) * scale
    bias = jax.random.normal(k_b, (O,), dtype) * scale
    return x, adj, weight, bias


def workload(x, adj, weight, bias):
    idx = jnp.arange(adj.shape[-1])
    adj = adj.at[:, idx, idx].set(CONFIG['loop_value'])
    out = jnp.matmul(adj, x)
    degree = jnp.sum(adj, axis=-1, keepdims=True)
    out = out / jnp.maximum(degree, CONFIG['min_degree'])
    out = jnp.matmul(out, weight)
    return out + bias
