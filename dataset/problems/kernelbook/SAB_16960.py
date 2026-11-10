"""SAB (set attention block): linear Q/K/V -> multi-head split -> softmax dot-product attention -> residual on Q -> head concat -> residual relu linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SAB_16960',
    'batch': 16,
    'set_size': 1024,
    'dim_in': 1024,
    'dim_out': 1024,
    'num_heads': 4,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    b, n = CONFIG['batch'], CONFIG['set_size']
    d_in, d_out = CONFIG['dim_in'], CONFIG['dim_out']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(keys[0], (b, n, d_in), dtype)
    w_q = jax.random.normal(keys[1], (d_in, d_out), dtype) * scale
    b_q = jax.random.normal(keys[2], (d_out,), dtype) * scale
    w_k = jax.random.normal(keys[3], (d_in, d_out), dtype) * scale
    b_k = jax.random.normal(keys[4], (d_out,), dtype) * scale
    w_v = jax.random.normal(keys[5], (d_in, d_out), dtype) * scale
    b_v = jax.random.normal(keys[6], (d_out,), dtype) * scale
    w_o = jax.random.normal(keys[7], (d_out, d_out), dtype) * scale
    b_o = jax.random.normal(keys[8], (d_out,), dtype) * scale
    return x, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o


def _split_heads(x, num_heads):
    b, n, v = x.shape
    return jnp.transpose(x.reshape(b, n, num_heads, v // num_heads), (2, 0, 1, 3))


def workload(x, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o):
    num_heads = CONFIG['num_heads']
    dim_v = w_q.shape[1]
    q = x @ w_q + b_q
    k = x @ w_k + b_k
    v = x @ w_v + b_v
    q_ = _split_heads(q, num_heads)
    k_ = _split_heads(k, num_heads)
    v_ = _split_heads(v, num_heads)
    a = jax.nn.softmax(q_ @ jnp.swapaxes(k_, -1, -2) / (dim_v ** 0.5), axis=-1)
    o_ = q_ + a @ v_
    o = jnp.transpose(o_, (1, 2, 0, 3)).reshape(q.shape)
    return o + jax.nn.relu(o @ w_o + b_o)
