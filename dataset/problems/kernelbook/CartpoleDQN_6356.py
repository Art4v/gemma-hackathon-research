"""CartpoleDQN: Linear -> ReLU -> Linear on the last axis of a 4D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CartpoleDQN_6356',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'state_space_dim': 512,
    'hidden': 1024,
    'action_space_dim': 512,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['state_space_dim']),
        dtype,
    )
    w1 = jax.random.normal(kw1, (CONFIG['state_space_dim'], CONFIG['hidden']), dtype) * scale
    b1 = jax.random.normal(kb1, (CONFIG['hidden'],), dtype) * scale
    w2 = jax.random.normal(kw2, (CONFIG['hidden'], CONFIG['action_space_dim']), dtype) * scale
    b2 = jax.random.normal(kb2, (CONFIG['action_space_dim'],), dtype) * scale
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jnp.maximum(x @ w1 + b1, 0)
    return h @ w2 + b2
