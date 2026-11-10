"""SACActorNetwork: squeeze(1) -> Linear -> ReLU -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SACActorNetwork_1536',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'n_input': 128,
    'n_features': 256,
    'n_output': 128,
    'squeeze_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    state_shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['n_input'])
    state = jax.random.uniform(keys[0], state_shape, dtype)
    w1 = jax.random.normal(keys[1], (CONFIG['n_input'], CONFIG['n_features']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (CONFIG['n_features'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (CONFIG['n_features'], CONFIG['n_features']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (CONFIG['n_features'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (CONFIG['n_features'], CONFIG['n_output']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (CONFIG['n_output'],), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    axis = CONFIG['squeeze_axis']
    x = jnp.squeeze(state, axis) if state.shape[axis] == 1 else state
    x = x.astype(w1.dtype)
    features1 = jax.nn.relu(x @ w1 + b1)
    features2 = jax.nn.relu(features1 @ w2 + b2)
    return features2 @ w3 + b3
