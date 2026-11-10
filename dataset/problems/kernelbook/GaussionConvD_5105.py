"""GaussionConvD: Linear -> ELU / Linear -> ReLU, exp(-gamma*var) attention, adjacency matmuls for mean and var."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussionConvD_5105',
    'num_nodes': 4096,
    'in_features': 1024,
    'out_features': 1024,
    'gamma': 1.0,
}


def create_inputs(dtype=jnp.float32):
    n = CONFIG['num_nodes']
    fin = CONFIG['in_features']
    fout = CONFIG['out_features']
    keys = jax.random.split(jax.random.key(0), 6)
    mean = jax.random.uniform(keys[0], (n, fin), dtype)
    var = jax.random.uniform(keys[1], (n, fin), dtype)
    adj_mean = jax.random.uniform(keys[2], (n, n), dtype)
    adj_var = jax.random.uniform(keys[3], (n, n), dtype)
    w_mean = jax.random.normal(keys[4], (fin, fout), dtype) * 0.02
    w_var = jax.random.normal(keys[5], (fin, fout), dtype) * 0.02
    return mean, var, adj_mean, adj_var, w_mean, w_var


def workload(mean, var, adj_mean, adj_var, w_mean, w_var):
    gamma = CONFIG['gamma']
    mean = jax.nn.elu(mean @ w_mean)
    var = jax.nn.relu(var @ w_var)
    attention = jnp.exp(-gamma * var)
    out_mean = adj_mean @ (mean * attention)
    out_var = adj_var @ (var * attention * attention)
    return out_mean, out_var
