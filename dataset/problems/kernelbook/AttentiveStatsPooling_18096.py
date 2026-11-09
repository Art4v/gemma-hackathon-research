"""Attentive stats pooling: linear -> tanh -> linear -> softmax over time -> weighted mean and std -> concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentiveStatsPooling_18096',
    'batch': 32,
    'input_size': 512,
    'time': 2048,
    'hidden_size': 128,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.float32):
    kx, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    b, de, t, h = CONFIG['batch'], CONFIG['input_size'], CONFIG['time'], CONFIG['hidden_size']
    x = jax.random.uniform(kx, (b, de, t), dtype)
    w_in = jax.random.normal(k1, (de, h), dtype) * 0.02
    b_in = jax.random.normal(k2, (h,), dtype) * 0.02
    w_out = jax.random.normal(k3, (h, de), dtype) * 0.02
    b_out = jax.random.normal(k4, (de,), dtype) * 0.02
    return x, w_in, b_in, w_out, b_out


def workload(x, w_in, b_in, w_out, b_out):
    xt = jnp.swapaxes(x, 1, 2)
    energies = jnp.swapaxes(jnp.tanh(xt @ w_in + b_in) @ w_out + b_out, 1, 2)
    alphas = jax.nn.softmax(energies, axis=2)
    means = jnp.sum(alphas * x, axis=2)
    residuals = jnp.sum(alphas * x ** 2, axis=2) - means ** 2
    stds = jnp.sqrt(jnp.maximum(residuals, CONFIG['eps']))
    return jnp.concatenate([means, stds], axis=1)
