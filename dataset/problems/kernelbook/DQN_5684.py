"""DQN forward: Linear -> ReLU -> Linear applied over the last axis of a rank-4 input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DQN_5684',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'state_dim': 256,
    'hidden_dim': 512,
    'out_dim': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['state_dim']),
        dtype,
    )
    w1 = jax.random.normal(kw1, (CONFIG['state_dim'], CONFIG['hidden_dim']), dtype) * scale
    b1 = jax.random.normal(kb1, (CONFIG['hidden_dim'],), dtype) * scale
    w2 = jax.random.normal(kw2, (CONFIG['hidden_dim'], CONFIG['out_dim']), dtype) * scale
    b2 = jax.random.normal(kb2, (CONFIG['out_dim'],), dtype) * scale
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
