"""HGCN_6507: hypergraph convolution D^-1/2 H |W| B^-1 H^T D^-1/2 X with full diagonal matrices and inf-to-zero inverse cleanup."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HGCN_6507',
    'batch': 16,
    'n_agents': 1024,
    'n_edges': 1024,
    'feature_dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_features, k_graph = jax.random.split(key)
    batch = CONFIG['batch']
    n_agents = CONFIG['n_agents']
    n_edges = CONFIG['n_edges']
    feature_dim = CONFIG['feature_dim']
    node_features = jax.random.uniform(k_features, (batch, n_agents, feature_dim), dtype)
    hyper_graph = jax.random.uniform(k_graph, (batch, n_agents, n_edges), dtype)
    W_line = jnp.ones((n_edges,), dtype)
    return node_features, hyper_graph, W_line


def workload(node_features, hyper_graph, W_line):
    diag_embed = jax.vmap(jnp.diag)
    abs_W = jnp.abs(jnp.diag(W_line))

    B_inv = diag_embed(jnp.sum(hyper_graph, axis=-2))
    D_inv = diag_embed(jnp.sum(jnp.matmul(hyper_graph, abs_W), axis=-1))

    # Elementwise over the full matrices: off-diagonal zeros become inf, then 0.
    D_inv = 1.0 / jnp.sqrt(D_inv)
    B_inv = 1.0 / B_inv
    # PyTorch's `== nan` checks never match, so only +inf is replaced.
    D_inv = jnp.where(D_inv == jnp.inf, jnp.zeros_like(D_inv), D_inv)
    B_inv = jnp.where(B_inv == jnp.inf, jnp.zeros_like(B_inv), B_inv)

    A = jnp.matmul(D_inv, hyper_graph)
    A = jnp.matmul(A, abs_W)
    A = jnp.matmul(A, B_inv)
    A = jnp.matmul(A, jnp.swapaxes(hyper_graph, -2, -1))
    A = jnp.matmul(A, D_inv)
    return jnp.matmul(A, node_features)
