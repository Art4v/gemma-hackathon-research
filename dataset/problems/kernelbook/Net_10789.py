"""[bf16 variant] Net_10789: 6-layer MLP, (Linear -> ReLU) x5 -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net_10789',
    'x_shape': (16, 16, 128, 1024),
    'n_feature': 1024,
    'n_hidden1': 1024,
    'n_hidden2': 1024,
    'n_hidden3': 1024,
    'n_hidden4': 1024,
    'n_hidden5': 1024,
    'n_output': 1024,
}

_DIM_KEYS = ('n_feature', 'n_hidden1', 'n_hidden2', 'n_hidden3',
             'n_hidden4', 'n_hidden5', 'n_output')


def create_inputs(dtype=jnp.bfloat16):
    dims = [CONFIG[k] for k in _DIM_KEYS]
    n_layers = len(dims) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_layers)
    x = jax.random.uniform(keys[0], CONFIG['x_shape'], dtype)
    params = []
    for i in range(n_layers):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def workload(x, *params):
    n_layers = len(params) // 2
    for i in range(n_layers):
        w, b = params[2 * i], params[2 * i + 1]
        x = x @ w + b
        if i < n_layers - 1:
            x = jax.nn.relu(x)
    return x
