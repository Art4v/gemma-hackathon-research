"""DeeperAutoencoder: 6x Linear with ReLU after the first five (dense autoencoder MLP chain)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DeeperAutoencoder_10568',
    'batch_shape': (32, 32, 64),
    'input_length': 384,
    'output_length': 384,
    'neuron_multiplier': 1,
    'hidden': 4 * 256,
    'bottleneck': 256,
    'expand': int(384 / 3) + 384,
    'sigmoid': False,
}


def create_inputs(dtype=jnp.float32):
    L = CONFIG['input_length']
    O = CONFIG['output_length']
    m = CONFIG['neuron_multiplier']
    H = CONFIG['hidden'] * m
    B = CONFIG['bottleneck'] * m
    E = CONFIG['expand']
    dims = [(L, H), (H, B), (B, H), (H, O), (O, E), (E, O)]
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(dims))
    x = jax.random.uniform(keys[0], (*CONFIG['batch_shape'], L), dtype)
    params = []
    for i, (fan_in, fan_out) in enumerate(dims):
        params.append(jax.random.normal(keys[1 + 2 * i], (fan_in, fan_out), dtype) * 0.02)
        params.append(jax.random.normal(keys[2 + 2 * i], (fan_out,), dtype) * 0.02)
    return (x, *params)


def workload(x, w1, b1, w2, b2, w5, b5, w6, b6, w7, b7, wo, bo):
    for w, b in ((w1, b1), (w2, b2), (w5, b5), (w6, b6), (w7, b7)):
        x = jax.nn.relu(x @ w + b)
    x = x @ wo + bo
    if CONFIG['sigmoid']:
        x = jax.nn.sigmoid(x)
    return x
