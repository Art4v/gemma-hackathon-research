"""[bf16 variant] Linear -> ReLU -> Linear (NeuralNetPartialNoGradModel MLP over the last axis)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NeuralNetPartialNoGradModel_3264',
    'lead_shape': (8, 8, 512),
    'input_size': 1024,
    'hidden_size': 1024,
    'num_classes': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    d_in, d_h, d_out = CONFIG['input_size'], CONFIG['hidden_size'], CONFIG['num_classes']
    x = jax.random.uniform(k_x, (*CONFIG['lead_shape'], d_in), dtype)
    w1 = jax.random.normal(k_w1, (d_in, d_h), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (d_h,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (d_h, d_out), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (d_out,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jnp.maximum(x @ w1 + b1, 0)
    return h @ w2 + b2
