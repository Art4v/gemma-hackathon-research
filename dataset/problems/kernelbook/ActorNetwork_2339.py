"""ActorNetwork: Linear -> LayerNorm -> ReLU -> Linear -> ReLU -> Linear -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ActorNetwork_2339',
    'input_shape': (32, 16, 128),
    'state_dim': 512,
    'fc1_units': 1024,
    'fc2_units': 512,
    'action_dim': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    s, f1, f2, a = CONFIG['state_dim'], CONFIG['fc1_units'], CONFIG['fc2_units'], CONFIG['action_dim']
    state = jax.random.uniform(keys[0], CONFIG['input_shape'] + (s,), dtype)
    w1 = jax.random.normal(keys[1], (s, f1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (f1,), dtype) * 0.02
    gamma = jnp.ones((f1,), dtype)
    beta = jnp.zeros((f1,), dtype)
    w2 = jax.random.normal(keys[3], (f1, f2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (f2,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (f2, a), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (a,), dtype) * 0.02
    return state, w1, b1, gamma, beta, w2, b2, w3, b3


def workload(state, w1, b1, gamma, beta, w2, b2, w3, b3):
    h = state @ w1 + b1
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(h - mean), axis=-1, keepdims=True)
    h = (h - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta
    h = jax.nn.relu(h)
    h = jax.nn.relu(h @ w2 + b2)
    return jnp.tanh(h @ w3 + b3)
