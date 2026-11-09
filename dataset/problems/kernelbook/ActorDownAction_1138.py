"""[bf16 variant] ActorDownAction: concat -> tanh -> two 3-layer ReLU MLPs; max_action*tanh action head and L2-normalized message head."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ActorDownAction_1138',
    'batch': 64,
    'num_nodes': 2048,
    'self_input_dim': 128,
    'msg_dim': 128,
    'action_dim': 128,
    'max_children': 4,
    'hidden1': 400,
    'hidden2': 300,
    'max_action': 4.0,
    'normalize_eps': 1e-12,
}


def _mlp_params(keys, d_in, d_out, dtype):
    h1, h2 = CONFIG['hidden1'], CONFIG['hidden2']
    dims = [(d_in, h1), (h1, h2), (h2, d_out)]
    params = []
    for (a, b), kw, kb in zip(dims, keys[0::2], keys[1::2]):
        params.append(jax.random.normal(kw, (a, b), dtype) * 0.02)
        params.append(jax.random.normal(kb, (b,), dtype) * 0.02)
    return params


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 14)
    lead = (CONFIG['batch'], CONFIG['num_nodes'])
    x = jax.random.uniform(keys[0], lead + (CONFIG['self_input_dim'],), dtype)
    m = jax.random.uniform(keys[1], lead + (CONFIG['msg_dim'],), dtype)
    d_in = CONFIG['self_input_dim'] + CONFIG['msg_dim']
    action_params = _mlp_params(keys[2:8], d_in, CONFIG['action_dim'], dtype)
    msg_params = _mlp_params(keys[8:14], d_in, CONFIG['msg_dim'] * CONFIG['max_children'], dtype)
    return (x, m, *action_params, *msg_params)


def _mlp(h, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(h @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3


def workload(x, m, aw1, ab1, aw2, ab2, aw3, ab3, mw1, mb1, mw2, mb2, mw3, mb3):
    xm = jnp.tanh(jnp.concatenate([x, m], axis=-1))
    action = CONFIG['max_action'] * jnp.tanh(_mlp(xm, aw1, ab1, aw2, ab2, aw3, ab3))
    msg = _mlp(xm, mw1, mb1, mw2, mb2, mw3, mb3)
    norm = jnp.sqrt(jnp.sum(msg * msg, axis=-1, keepdims=True))
    msg_down = msg / jnp.maximum(norm, CONFIG['normalize_eps'])
    return action, msg_down
