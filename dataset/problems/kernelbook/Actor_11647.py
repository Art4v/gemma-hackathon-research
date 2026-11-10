"""[bf16 variant] Actor MLP: Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh -> scale by action_range."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_11647',
    'batch': 16,
    'dim1': 128,
    'dim2': 256,
    'state_dim': 128,
    'hidden_dim': 16,
    'action_dim': 128,
    'action_range': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    s, h, a = CONFIG['state_dim'], CONFIG['hidden_dim'], CONFIG['action_dim']
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], s), dtype)
    w1 = jax.random.normal(keys[1], (s, h), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h, h), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h, a), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (a,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return jnp.tanh(h @ w3 + b3) * CONFIG['action_range']
