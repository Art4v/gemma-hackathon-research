"""Net_11161: reshape -> Linear(784,64) -> ReLU -> Linear(64,10)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net_11161',
    'batch': 65536,
    'in_features': 784,
    'hidden': 64,
    'out_features': 10,
}


def create_inputs(dtype=jnp.float32):
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    b, d_in = CONFIG['batch'], CONFIG['in_features']
    h, d_out = CONFIG['hidden'], CONFIG['out_features']
    x = jax.random.uniform(kx, (b, d_in), dtype)
    w1 = jax.random.normal(k1, (d_in, h), dtype) * 0.02
    b1 = jax.random.normal(kb1, (h,), dtype) * 0.02
    w2 = jax.random.normal(k2, (h, d_out), dtype) * 0.02
    b2 = jax.random.normal(kb2, (d_out,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    x = x.reshape(-1, CONFIG['in_features'])
    x = jax.nn.relu(x @ w1 + b1)
    return x @ w2 + b2
