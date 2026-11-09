"""[bf16 variant] GraphConvolution: matmul(nodes, adj) -> LeakyReLU(0.2) -> 1x1 Conv1d -> LeakyReLU(0.2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GraphConvolution_8391',
    'batch': 8,
    'in_dim': 512,
    'out_dim': 512,
    'num_nodes': 4096,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.bfloat16):
    k_adj, k_nodes, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    n = CONFIG['num_nodes']
    adj = jax.random.uniform(k_adj, (n, n), dtype)
    nodes = jax.random.uniform(k_nodes, (CONFIG['batch'], CONFIG['in_dim'], n), dtype)
    w = jax.random.normal(k_w, (CONFIG['in_dim'], CONFIG['out_dim']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_dim'],), dtype) * 0.02
    return adj, nodes, w, b


def workload(adj, nodes, w, b):
    slope = CONFIG['negative_slope']
    h = jax.nn.leaky_relu(jnp.matmul(nodes, adj), negative_slope=slope)
    # 1x1 Conv1d over (..., C_in, N): channel mix with w stored (in, out).
    h = jnp.einsum('co,...cn->...on', w, h) + b[:, None]
    return jax.nn.leaky_relu(h, negative_slope=slope)
