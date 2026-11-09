"""GatedTanh: tanh(x @ W + b) * sigmoid(x @ W_gate + b_gate)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GatedTanh_18083',
    'batch': 16,
    'seq_len': 1024,
    'in_dim': 2048,
    'out_dim': 2048,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b, k_wg, k_bg = jax.random.split(jax.random.key(0), 5)
    in_dim, out_dim = CONFIG['in_dim'], CONFIG['out_dim']
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['seq_len'], in_dim), dtype)
    w_fc = jax.random.normal(k_w, (in_dim, out_dim), dtype) * 0.02
    b_fc = jax.random.normal(k_b, (out_dim,), dtype) * 0.02
    w_gate = jax.random.normal(k_wg, (in_dim, out_dim), dtype) * 0.02
    b_gate = jax.random.normal(k_bg, (out_dim,), dtype) * 0.02
    return x, w_fc, b_fc, w_gate, b_gate


def workload(x, w_fc, b_fc, w_gate, b_gate):
    y_tilda = jnp.tanh(x @ w_fc + b_fc)
    gate = jax.nn.sigmoid(x @ w_gate + b_gate)
    return y_tilda * gate
