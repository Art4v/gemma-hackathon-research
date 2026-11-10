"""[bf16 variant] Twin-Q critic: concat(state, action) -> two MLPs (Linear-ReLU-Linear-ReLU-Linear->1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_2607',
    'batch': 8192,
    'state_size': 1024,
    'action_size': 1024,
    'fc1_units': 2048,
    'fc2_units': 1536,
    'out_units': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    b, s, a = c['batch'], c['state_size'], c['action_size']
    d_in, h1, h2, o = s + a, c['fc1_units'], c['fc2_units'], c['out_units']
    keys = jax.random.split(jax.random.key(0), 14)
    state = jax.random.uniform(keys[0], (b, s), dtype)
    action = jax.random.uniform(keys[1], (b, a), dtype)
    shapes = [(d_in, h1), (h1,), (h1, h2), (h2,), (h2, o), (o,)] * 2
    params = tuple(
        jax.random.normal(k, shp, dtype) * 0.02 for k, shp in zip(keys[2:], shapes)
    )
    return (state, action) + params


def _q_head(xa, w_a, b_a, w_b, b_b, w_c, b_c):
    h = jax.nn.relu(xa @ w_a + b_a)
    h = jax.nn.relu(h @ w_b + b_b)
    return h @ w_c + b_c


def workload(state, action, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):
    xa = jnp.concatenate([state, action], axis=1)
    q1 = _q_head(xa, w1, b1, w2, b2, w3, b3)
    q2 = _q_head(xa, w4, b4, w5, b5, w6, b6)
    return q1, q2
