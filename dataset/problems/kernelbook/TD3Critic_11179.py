"""TD3 twin critic: concat(state, action) -> two independent (Linear-ReLU-Linear-ReLU-Linear) Q heads."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TD3Critic_11179',
    'batch': 8192,
    'state_dim': 1024,
    'action_dim': 1024,
    'hidden_dim': 256,
    'output_dim': 1,
    'concat_axis': 1,
    'param_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    batch = CONFIG['batch']
    state_dim = CONFIG['state_dim']
    action_dim = CONFIG['action_dim']
    hidden = CONFIG['hidden_dim']
    output = CONFIG['output_dim']
    scale = CONFIG['param_scale']
    in_dim = state_dim + action_dim

    head_shapes = [
        (in_dim, hidden), (hidden,),
        (hidden, hidden), (hidden,),
        (hidden, output), (output,),
    ]
    param_shapes = head_shapes + head_shapes

    keys = jax.random.split(jax.random.key(0), len(param_shapes) + 2)
    state = jax.random.uniform(keys[0], (batch, state_dim), dtype)
    action = jax.random.uniform(keys[1], (batch, action_dim), dtype)
    params = tuple(
        jax.random.normal(keys[i + 2], shape, dtype) * scale
        for i, shape in enumerate(param_shapes)
    )
    return (state, action) + params


def _q_head(sa, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(sa @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3


def workload(state, action,
             w1, b1, w2, b2, w3, b3,
             w4, b4, w5, b5, w6, b6):
    sa = jnp.concatenate([state, action], axis=CONFIG['concat_axis'])
    q1 = _q_head(sa, w1, b1, w2, b2, w3, b3)
    q2 = _q_head(sa, w4, b4, w5, b5, w6, b6)
    return q1, q2
