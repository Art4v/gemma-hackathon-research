"""Linear -> MaxPool1d -> channel select -> Linear (LinearMaxPoolLinearModel_436)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LinearMaxPoolLinearModel_436',
    'batch': 4096,
    'in_features': 4096,
    'pool_size': 4,
    'pooled_features': 1024,
    'out_features': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['in_features']), dtype)
    w1 = jax.random.normal(k_w1, (CONFIG['in_features'], CONFIG['in_features']), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (CONFIG['pooled_features'], CONFIG['out_features']), dtype) * 0.02
    return x, w1, w2


def workload(x, w1, w2):
    k = CONFIG['pool_size']
    h = x[:, None, :] @ w1
    pooled = lax.reduce_window(h, -jnp.inf, lax.max, (1, 1, k), (1, 1, k), 'VALID')
    return pooled[:, 0, :] @ w2
