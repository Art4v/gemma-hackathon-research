"""[bf16 variant] Critic_10668: concat(state, action) -> twin MLP heads (Linear-ReLU-Linear-ReLU-Linear) -> (q1, q2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_10668',
    'batch': 65536,
    'state_dim': 512,
    'action_dim': 512,
    'hidden1': 6,
    'hidden2': 4,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    d_in = CONFIG['state_dim'] + CONFIG['action_dim']
    h1, h2, out = CONFIG['hidden1'], CONFIG['hidden2'], CONFIG['out_dim']
    keys = jax.random.split(jax.random.key(0), 14)
    state = jax.random.uniform(keys[0], (b, CONFIG['state_dim']), dtype)
    action = jax.random.uniform(keys[1], (b, CONFIG['action_dim']), dtype)
    shapes = [(d_in, h1), (h1,), (h1, h2), (h2,), (h2, out), (out,)] * 2
    params = tuple(
        jax.random.normal(k, s, dtype) * 0.02 for k, s in zip(keys[2:], shapes)
    )
    return (state, action) + params


def _head(sa, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(sa @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3


def workload(state, action, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):
    sa = jnp.concatenate([state, action], axis=1)
    q1 = _head(sa, w1, b1, w2, b2, w3, b3)
    q2 = _head(sa, w4, b4, w5, b5, w6, b6)
    return q1, q2
