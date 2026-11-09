"""Value_Net: concat(state, action) -> Linear -> ReLU -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Value_Net_136',
    'batch': 32768,
    'observation_dim': 512,
    'action_dim': 512,
    'hidden': 256,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    obs, act = CONFIG['observation_dim'], CONFIG['action_dim']
    h, o = CONFIG['hidden'], CONFIG['out_dim']
    ks = jax.random.split(jax.random.key(0), 8)
    state = jax.random.uniform(ks[0], (b, obs), dtype)
    action = jax.random.uniform(ks[1], (b, act), dtype)
    w1 = jax.random.normal(ks[2], (obs + act, h), dtype) * 0.02
    b1 = jax.random.normal(ks[3], (h,), dtype) * 0.02
    w2 = jax.random.normal(ks[4], (h, h), dtype) * 0.02
    b2 = jax.random.normal(ks[5], (h,), dtype) * 0.02
    w3 = jax.random.normal(ks[6], (h, o), dtype) * 0.02
    b3 = jax.random.normal(ks[7], (o,), dtype) * 0.02
    return state, action, w1, b1, w2, b2, w3, b3


def workload(state, action, w1, b1, w2, b2, w3, b3):
    x = jnp.concatenate((state, action), axis=1)
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
