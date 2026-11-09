"""GaussionConvD: Linear -> ELU / Linear -> ReLU, exp(-gamma*var) attention, adjacency matmuls (RobustGCN layer)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussionConvD_13636',
    'num_nodes': 4096,
    'in_features': 1024,
    'out_features': 1024,
    'gamma': 1.0,
}


def create_inputs(dtype=jnp.float32):
    n = CONFIG['num_nodes']
    fi = CONFIG['in_features']
    fo = CONFIG['out_features']
    k = jax.random.split(jax.random.key(0), 6)
    mean = jax.random.uniform(k[0], (n, fi), dtype)
    var = jax.random.uniform(k[1], (n, fi), dtype)
    adj_mean = jax.random.uniform(k[2], (n, n), dtype)
    adj_var = jax.random.uniform(k[3], (n, n), dtype)
    w_mean = jax.random.normal(k[4], (fi, fo), dtype) * 0.02
    w_var = jax.random.normal(k[5], (fi, fo), dtype) * 0.02
    return mean, var, adj_mean, adj_var, w_mean, w_var


def workload(mean, var, adj_mean, adj_var, w_mean, w_var):
    m = jax.nn.elu(mean @ w_mean)
    v = jax.nn.relu(var @ w_var)
    attention = jnp.exp(-CONFIG['gamma'] * v)
    out_mean = adj_mean @ (m * attention)
    out_var = adj_var @ (v * attention * attention)
    return out_mean, out_var
