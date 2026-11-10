"""[bf16 variant] AE_2D_v1000: Linear x8 (in-1024-384-128-2-128-384-1024-out) with tanh between consecutive layers."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AE_2D_v1000_12448',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'n_features': 256,
    'hidden1': 1024,
    'hidden2': 384,
    'hidden3': 128,
    'latent': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    n = CONFIG['n_features']
    h1 = CONFIG['hidden1']
    h2 = CONFIG['hidden2']
    h3 = CONFIG['hidden3']
    z = CONFIG['latent']
    dims = (n, h1, h2, h3, z, h3, h2, h1, n)
    n_layers = len(dims) - 1

    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_layers)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], n),
        dtype,
    )

    params = []
    for i in range(n_layers):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def workload(*inputs):
    h = inputs[0]
    params = inputs[1:]
    for i in range(0, len(params), 2):
        if i > 0:
            h = jnp.tanh(h)
        h = h @ params[i] + params[i + 1]
    return h
