"""PGNet: Linear + bias -> ReLU -> Linear + bias -> Softmax(axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PGNet_9289',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'n_features': 256,
    'hidden': 256,
    'n_actions': 256,
    'weight_scale': 0.02,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_features']),
        dtype,
    )
    w1 = jax.random.normal(kw1, (CONFIG['n_features'], CONFIG['hidden']), dtype) * scale
    b1 = jax.random.normal(kb1, (CONFIG['hidden'],), dtype) * scale
    w2 = jax.random.normal(kw2, (CONFIG['hidden'], CONFIG['n_actions']), dtype) * scale
    b2 = jax.random.normal(kb2, (CONFIG['n_actions'],), dtype) * scale
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jnp.maximum(x @ w1 + b1, 0)
    logits = h @ w2 + b2
    return jax.nn.softmax(logits, axis=CONFIG['softmax_axis'])
