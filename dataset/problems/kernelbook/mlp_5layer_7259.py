"""[bf16 variant] mlp_5layer: flatten -> (Linear+ReLU) x4 -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mlp_5layer_7259',
    'batch': 2048,
    'in_ch': 4,
    'in_dim': 64,
    'width': 1,
    'hidden': 256,
    'hidden4': 128,
    'out_features': 10,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    in_f = c['in_ch'] * c['in_dim'] * c['in_dim']
    h = c['hidden'] * c['width']
    h4 = c['hidden4'] * c['width']
    dims = [in_f, h, h, h, h4, c['out_features']]
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_ch'], c['in_dim'], c['in_dim']), dtype)
    params = []
    for i in range(5):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    h = x.reshape(x.shape[0], -1)
    h = jax.nn.relu(h @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    h = jax.nn.relu(h @ w3 + b3)
    h = jax.nn.relu(h @ w4 + b4)
    return h @ w5 + b5
