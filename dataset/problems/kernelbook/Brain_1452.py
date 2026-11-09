"""Linear(4->128) -> ReLU -> {Linear(128->2) -> softmax, Linear(128->1)} (A2C actor-critic Brain forward)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Brain_1452',
    'batch': 8,
    'dim1': 512,
    'dim2': 1024,
    'in_features': 4,
    'hidden': 128,
    'num_actions': 2,
    'value_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(k[0], (c['batch'], c['dim1'], c['dim2'], c['in_features']), dtype)
    w1 = jax.random.normal(k[1], (c['in_features'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c['hidden'],), dtype) * 0.02
    wa = jax.random.normal(k[3], (c['hidden'], c['num_actions']), dtype) * 0.02
    ba = jax.random.normal(k[4], (c['num_actions'],), dtype) * 0.02
    wv = jax.random.normal(k[5], (c['hidden'], c['value_dim']), dtype) * 0.02
    bv = jax.random.normal(k[6], (c['value_dim'],), dtype) * 0.02
    return x, w1, b1, wa, ba, wv, bv


def workload(x, w1, b1, wa, ba, wv, bv):
    h = jax.nn.relu(x @ w1 + b1)
    probs = jax.nn.softmax(h @ wa + ba, axis=-1)
    value = h @ wv + bv
    return probs, value
