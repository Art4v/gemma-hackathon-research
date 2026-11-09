"""DenseSAGEConv: self-loop adj, adj @ x, mean by degree, linear + bias, L2 normalize."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DenseSAGEConv_6086',
    'num_adj': 8,
    'num_nodes': 256,
    'in_channels': 256,
    'out_channels': 256,
    'norm_eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    kx, ka, kw, kb = jax.random.split(jax.random.key(0), 4)
    n = CONFIG['num_nodes']
    # KernelBench passes a 4D adj (A, N, N, N) that broadcasts against x (N, N, F).
    x = jax.random.uniform(kx, (n, n, CONFIG['in_channels']), dtype)
    adj = jax.random.uniform(ka, (CONFIG['num_adj'], n, n, n), dtype)
    weight = jax.random.normal(kw, (CONFIG['in_channels'], CONFIG['out_channels']), dtype) * 0.02
    bias = jax.random.normal(kb, (CONFIG['out_channels'],), dtype) * 0.02
    return x, adj, weight, bias


def workload(x, adj, weight, bias):
    n = x.shape[-2]
    idx = jnp.arange(n)
    # Same as torch adj[:, idx, idx] = 1: diagonal over dims 1 and 2.
    adj = adj.at[:, idx, idx].set(1.0)
    out = jnp.matmul(adj, x)
    out = out / jnp.sum(adj, axis=-1, keepdims=True)
    out = jnp.matmul(out, weight) + bias
    norm = jnp.sqrt(jnp.sum(out * out, axis=-1, keepdims=True))
    return out / jnp.maximum(norm, CONFIG['norm_eps'])
