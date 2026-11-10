"""ResidualLinear: Linear -> ReLU -> Linear -> residual add -> ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResidualLinear_14910',
    'batch': 8,
    'seq_len': 2048,
    'hidden_dim': 2048,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    b, s, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_dim']
    x = jax.random.uniform(k_x, (b, s, h), dtype)
    w1 = jax.random.normal(k_w1, (h, h), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (h,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (h, h), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (h,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    # norm1 / norm2 default to None in the source module, so no normalization.
    f = jax.nn.relu(x @ w1 + b1)
    f = f @ w2 + b2
    return jax.nn.relu(f + x)
