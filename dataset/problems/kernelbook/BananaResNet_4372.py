"""[bf16 variant] BananaResNet: 7-layer Linear+ReLU MLP with residual skip adds."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BananaResNet_4372',
    'batch': 32,
    'dim1': 64,
    'dim2': 128,
    'state_size': 128,
    'blk1_h1': 128,
    'blk1_h2': 128,
    'blk1_h3': 64,
    'blk2_h1': 64,
    'blk2_h2': 64,
    'blk2_h3': 32,
    'action_size': 128,
}


def _layer_dims():
    c = CONFIG
    return [
        (c['state_size'], c['blk1_h1']),
        (c['blk1_h1'], c['blk1_h2']),
        (c['blk1_h2'], c['blk1_h3']),
        (c['blk1_h3'], c['blk2_h1']),
        (c['blk2_h1'], c['blk2_h2']),
        (c['blk2_h2'], c['blk2_h3']),
        (c['blk2_h3'], c['action_size']),
    ]


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    dims = _layer_dims()
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(dims))
    state = jax.random.uniform(
        keys[0], (c['batch'], c['dim1'], c['dim2'], c['state_size']), dtype)
    params = []
    for i, (fin, fout) in enumerate(dims):
        params.append(jax.random.normal(keys[1 + 2 * i], (fin, fout), dtype) * 0.02)
        params.append(jax.random.normal(keys[2 + 2 * i], (fout,), dtype) * 0.02)
    return (state, *params)


def workload(state, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6, w7, b7):
    relu = jax.nn.relu
    skip = relu(state @ w1 + b1)
    x = relu(skip @ w2 + b2)
    x = x + skip
    skip = relu(x @ w3 + b3)
    x = relu(skip @ w4 + b4)
    skip = x + skip
    x = relu(x @ w5 + b5)
    skip = x + skip
    x = relu(skip @ w6 + b6)
    return x @ w7 + b7
