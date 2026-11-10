"""Actor MLP: Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU -> Linear -> tanh * 0.3."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_9191',
    'batch': 8,
    'seq1': 16,
    'seq2': 512,
    'num_inputs': 512,
    'hidden_size': 1024,
    'num_outputs': 512,  # must equal num_inputs: the module asserts mu.shape == inputs.shape
    'eps': 1e-5,
    'output_scale': 0.3,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(k[0], (c['batch'], c['seq1'], c['seq2'], c['num_inputs']), dtype)
    w1 = jax.random.normal(k[1], (c['num_inputs'], c['hidden_size']), dtype) * 0.02
    b1 = jnp.zeros((c['hidden_size'],), dtype)
    g1 = jnp.ones((c['hidden_size'],), dtype)
    be1 = jnp.zeros((c['hidden_size'],), dtype)
    w2 = jax.random.normal(k[2], (c['hidden_size'], c['hidden_size']), dtype) * 0.02
    b2 = jnp.zeros((c['hidden_size'],), dtype)
    g2 = jnp.ones((c['hidden_size'],), dtype)
    be2 = jnp.zeros((c['hidden_size'],), dtype)
    w3 = jax.random.normal(k[3], (c['hidden_size'], c['num_outputs']), dtype) * 0.02
    b3 = jnp.zeros((c['num_outputs'],), dtype)
    return x, w1, b1, g1, be1, w2, b2, g2, be2, w3, b3


def _layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, w1, b1, g1, be1, w2, b2, g2, be2, w3, b3):
    h = jax.nn.relu(_layer_norm(x @ w1 + b1, g1, be1))
    h = jax.nn.relu(_layer_norm(h @ w2 + b2, g2, be2))
    return jnp.tanh(h @ w3 + b3) * CONFIG['output_scale']
