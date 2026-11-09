"""PolicyNetwork: Linear -> ReLU -> Linear -> softmax over dim=1 (on a 4-D state, linear acts on last axis)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PolicyNetwork_3228',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'num_inputs': 128,
    'hidden_size': 256,
    'num_actions': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    state = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['num_inputs']), dtype)
    w1 = jax.random.normal(k_w1, (CONFIG['num_inputs'], CONFIG['hidden_size']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (CONFIG['hidden_size'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (CONFIG['hidden_size'], CONFIG['num_actions']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (CONFIG['num_actions'],), dtype) * 0.02
    return state, w1, b1, w2, b2


def workload(state, w1, b1, w2, b2):
    x = jax.nn.relu(state @ w1 + b1)
    logits = x @ w2 + b2
    # Matches torch F.softmax(..., dim=1): normalizes over axis 1, not the action axis.
    return jax.nn.softmax(logits, axis=CONFIG['softmax_axis'])
