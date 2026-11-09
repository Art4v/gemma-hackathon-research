"""Highway: relu(Linear_proj(x)) * sigmoid(Linear_gate(x)) + (1 - gate) * x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Highway_2434',
    'batch': 8,
    'dim1': 32,
    'dim2': 256,
    'e_word': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kwp, kbp, kwg, kbg = jax.random.split(key, 5)
    e = CONFIG['e_word']
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], e), dtype)
    w_proj = jax.random.normal(kwp, (e, e), dtype) * 0.02
    b_proj = jax.random.normal(kbp, (e,), dtype) * 0.02
    w_gate = jax.random.normal(kwg, (e, e), dtype) * 0.02
    b_gate = jax.random.normal(kbg, (e,), dtype) * 0.02
    return x, w_proj, b_proj, w_gate, b_gate


def workload(x, w_proj, b_proj, w_gate, b_gate):
    x_proj = jax.nn.relu(x @ w_proj + b_proj)
    x_gate = jax.nn.sigmoid(x @ w_gate + b_gate)
    return x_gate * x_proj + (1 - x_gate) * x
