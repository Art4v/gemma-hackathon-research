"""LR: Linear over last dim -> sum over dim 1 -> add bias -> unsqueeze(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LR_9130',
    'batch': 32,
    'dim1': 128,
    'dim2': 128,
    'feature_nums': 128,
    'output_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(k_x, (c['batch'], c['dim1'], c['dim2'], c['feature_nums']), dtype)
    w = jax.random.normal(k_w, (c['feature_nums'], c['output_dim']), dtype) * 0.02
    b_lin = jax.random.normal(k_b, (c['output_dim'],), dtype) * 0.02
    bias = jnp.zeros((c['output_dim'],), dtype)
    return x, w, b_lin, bias


def workload(x, w, b_lin, bias):
    y = jnp.matmul(x, w) + b_lin
    out = bias + jnp.sum(y, axis=1)
    return jnp.expand_dims(out, 1)
