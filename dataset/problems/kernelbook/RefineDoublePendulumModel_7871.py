"""SIREN autoencoder: 8 x (Linear -> sin(w0 * .)), last layer linear, returns (out, latent)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RefineDoublePendulumModel_7871',
    'batch': 16,
    'dim1': 64,
    'dim2': 128,
    'in_channels': 128,
    'hidden': (128, 64, 32, 4, 32, 64, 128),
    'latent_layer': 4,
    'w0': 30.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    dims = (c['in_channels'],) + tuple(c['hidden']) + (c['in_channels'],)
    n_layers = len(dims) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_layers)
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['in_channels']), dtype)
    params = []
    for i in range(n_layers):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def workload(x, *params):
    w0 = CONFIG['w0']
    n_layers = len(params) // 2
    latent = None
    for i in range(n_layers):
        x = x @ params[2 * i] + params[2 * i + 1]
        if i < n_layers - 1:
            x = jnp.sin(w0 * x)
        if i + 1 == CONFIG['latent_layer']:
            latent = x
    return x, latent
