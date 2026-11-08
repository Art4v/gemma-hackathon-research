"""Linear (matmul + bias) -> elementwise min with learnable constant -> subtract constant."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '68_Matmul_Min_Subtract',
    'batch_size': 1024,
    'in_features': 8192,
    'out_features': 8192,
    'constant': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch_size'], CONFIG['in_features']), dtype)
    # nn.Linear weight is (out, in) in PyTorch; stored here as (in, out).
    w = jax.random.normal(
        kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    c = jnp.asarray(CONFIG['constant'], dtype)
    return (x, w, b, c)


def workload(x, w, b, c):
    y = x @ w + b
    y = jnp.minimum(y, c)
    y = y - c
    return y
