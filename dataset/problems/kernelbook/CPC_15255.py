"""CPC score: Linear(y) -> L2 row-normalize(x_pred, x) -> pos dot + logsumexp(x @ x_pred.T) -> NCE mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CPC_15255',
    'batch': 8192,
    'x_size': 2048,
    'y_size': 1536,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, ky, kw, kb = jax.random.split(key, 4)
    B = CONFIG['batch']
    Dx = CONFIG['x_size']
    Dy = CONFIG['y_size']
    x = jax.random.uniform(kx, (B, Dx), dtype=dtype)
    y = jax.random.uniform(ky, (B, Dy), dtype=dtype)
    w = jax.random.normal(kw, (Dy, Dx), dtype=dtype) * 0.02
    b = jax.random.normal(kb, (Dx,), dtype=dtype) * 0.02
    return x, y, w, b


def workload(x, y, w, b):
    # net: Linear(y_size -> x_size); w is (in, out)
    x_pred = y @ w + b
    # L2 normalize along feature dim (torch .norm(dim=1), no eps)
    x_pred = x_pred / jnp.sqrt(jnp.sum(x_pred * x_pred, axis=1, keepdims=True))
    xn = x / jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    pos = jnp.sum(xn * x_pred, axis=-1)
    logits = xn @ x_pred.T
    neg = jax.nn.logsumexp(logits, axis=-1)
    nce = -jnp.mean(pos - neg)
    return nce
