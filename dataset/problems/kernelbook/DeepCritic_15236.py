"""[bf16 variant] DeepCritic: concat(state, action) -> Linear+ReLU, then 3x [concat skip -> Linear+ReLU] -> Linear(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DeepCritic_15236',
    'batch': 16384,
    'state_size': 1024,
    'action_size': 1024,
    'hidden_size': 1024,
    'out_size': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    s = CONFIG['state_size']
    a = CONFIG['action_size']
    h = CONFIG['hidden_size']
    o = CONFIG['out_size']
    in_dim = h + s + a
    keys = jax.random.split(jax.random.key(0), 12)
    state = jax.random.uniform(keys[0], (b, s), dtype)
    action = jax.random.uniform(keys[1], (b, a), dtype)
    w1 = jax.random.normal(keys[2], (s + a, h), dtype) * 0.02
    b1 = jax.random.normal(keys[3], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[4], (in_dim, h), dtype) * 0.02
    b2 = jax.random.normal(keys[5], (h,), dtype) * 0.02
    w3 = jax.random.normal(keys[6], (in_dim, h), dtype) * 0.02
    b3 = jax.random.normal(keys[7], (h,), dtype) * 0.02
    w4 = jax.random.normal(keys[8], (in_dim, h), dtype) * 0.02
    b4 = jax.random.normal(keys[9], (h,), dtype) * 0.02
    w5 = jax.random.normal(keys[10], (h, o), dtype) * 0.02
    b5 = jax.random.normal(keys[11], (o,), dtype) * 0.02
    return (state, action, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5)


def workload(state, action, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    xu = jnp.concatenate([state, action], axis=1)
    x = jax.nn.relu(xu @ w1 + b1)
    for w, b in ((w2, b2), (w3, b3), (w4, b4)):
        x = jax.nn.relu(jnp.concatenate([x, xu], axis=1) @ w + b)
    return x @ w5 + b5
