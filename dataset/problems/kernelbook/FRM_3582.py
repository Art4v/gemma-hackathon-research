"""FRM_3582: adaptive average pool over length -> linear -> sigmoid -> multiplicative and additive feature recalibration."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FRM_3582',
    'batch': 64,
    'nb_dim': 1024,
    'length': 1024,
    'weight_scale': 0.02,
    'do_mul': True,
    'do_add': True,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    batch = CONFIG['batch']
    nb_dim = CONFIG['nb_dim']
    length = CONFIG['length']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, (batch, nb_dim, length), dtype)
    w = jax.random.normal(k_w, (nb_dim, nb_dim), dtype) * scale
    b = jax.random.normal(k_b, (nb_dim,), dtype) * scale
    return x, w, b


def workload(x, w, b):
    y = jnp.mean(x, axis=-1)
    y = jax.nn.sigmoid(y @ w + b)
    y = jnp.reshape(y, (x.shape[0], x.shape[1], -1))
    if CONFIG['do_mul']:
        x = x * y
    if CONFIG['do_add']:
        x = x + y
    return x
