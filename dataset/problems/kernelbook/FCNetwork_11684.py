"""[bf16 variant] FCNetwork: Linear -> ReLU -> Linear -> ReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FCNetwork_11684',
    'input_shape': (16, 128, 128, 128),
    'state_size': 128,
    'hidden_size': 256,
    'action_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    s, h, a = CONFIG['state_size'], CONFIG['hidden_size'], CONFIG['action_size']
    shape = CONFIG['input_shape'][:-1] + (s,)
    x = jax.random.uniform(keys[0], shape, dtype)
    w1 = jax.random.normal(keys[1], (s, h), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (h, h), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (h,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (h, a), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (a,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    # ponytail: output_gate is None in the KernelBench init, so no gate
    return x @ w3 + b3
