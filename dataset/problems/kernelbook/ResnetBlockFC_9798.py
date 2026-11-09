"""ResnetBlockFC: x + fc_1(relu(fc_0(relu(x)))) with identity shortcut (size_in == size_out)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResnetBlockFC_9798',
    'batch': 8,
    'seq_len': 8192,
    'size_in': 1024,
    'size_h': 1024,
    'size_out': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w0, k_b0, k_w1, k_b1 = jax.random.split(jax.random.key(0), 5)
    d_in, d_h, d_out = CONFIG['size_in'], CONFIG['size_h'], CONFIG['size_out']
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['seq_len'], d_in), dtype)
    w0 = jax.random.normal(k_w0, (d_in, d_h), dtype) * 0.02
    b0 = jax.random.normal(k_b0, (d_h,), dtype) * 0.02
    w1 = jax.random.normal(k_w1, (d_h, d_out), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (d_out,), dtype) * 0.02
    return x, w0, b0, w1, b1


def workload(x, w0, b0, w1, b1):
    net = jax.nn.relu(x) @ w0 + b0
    dx = jax.nn.relu(net) @ w1 + b1
    # size_in == size_out, so the shortcut is the identity
    return x + dx
