"""FeedForward block: (x + branch) -> LayerNorm -> Linear -> ReLU -> Linear -> residual add -> LayerNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FeedForward_7227',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'ninp': 1024,
    'dim_feedforward': 4096,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 6)
    b, d1, d2 = CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2']
    ninp, dff = CONFIG['ninp'], CONFIG['dim_feedforward']
    x = jax.random.uniform(keys[0], (b, d1, d2, ninp), dtype)
    branch = jax.random.uniform(keys[1], (b, d1, d2, ninp), dtype)
    w1 = jax.random.normal(keys[2], (ninp, dff), dtype) * 0.02
    b1 = jax.random.normal(keys[3], (dff,), dtype) * 0.02
    w2 = jax.random.normal(keys[4], (dff, ninp), dtype) * 0.02
    b2 = jax.random.normal(keys[5], (ninp,), dtype) * 0.02
    g1 = jnp.ones((ninp,), dtype)
    be1 = jnp.zeros((ninp,), dtype)
    g2 = jnp.ones((ninp,), dtype)
    be2 = jnp.zeros((ninp,), dtype)
    return x, branch, w1, b1, w2, b2, g1, be1, g2, be2


def _layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, branch, w1, b1, w2, b2, g1, be1, g2, be2):
    # The PyTorch chunk loop over dim 1 has no effect on the result: every op here is per position.
    x = _layer_norm(x + branch, g1, be1)
    h = jnp.maximum(x @ w1 + b1, 0.0) @ w2 + b2
    return _layer_norm(x + h, g2, be2)
