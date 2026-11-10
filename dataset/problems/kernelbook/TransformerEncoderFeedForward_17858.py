"""[bf16 variant] TransformerEncoderFeedForward: Linear -> ReLU -> Linear -> (dropout=identity) -> residual add -> LayerNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TransformerEncoderFeedForward_17858',
    'batch': 8,
    'seq_len': 2048,
    'd_model': 1024,
    'expansion_rate': 4,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    b, s, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']
    hidden = d * CONFIG['expansion_rate']
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (b, s, d), dtype)
    w1 = jax.random.normal(k_w1, (d, hidden), dtype) * 0.02
    b1 = jnp.zeros((hidden,), dtype)
    w2 = jax.random.normal(k_w2, (hidden, d), dtype) * 0.02
    b2 = jnp.zeros((d,), dtype)
    gamma = jnp.ones((d,), dtype)
    beta = jnp.zeros((d,), dtype)
    return x, w1, b1, w2, b2, gamma, beta


def workload(x, w1, b1, w2, b2, gamma, beta):
    h = jax.nn.relu(x @ w1 + b1)
    y = x + (h @ w2 + b2)
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    return (y - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta
