"""[bf16 variant] ResBlock: pad -> Linear+CELU -> Linear+CELU * tanh(Linear) -> residual add -> LayerNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResBlock_7556',
    'batch': 64,
    'seq': 512,
    'input_size': 1024,
    'output_size': 1024,
    'celu_alpha': 1.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    b, s = CONFIG['batch'], CONFIG['seq']
    din, dout = CONFIG['input_size'], CONFIG['output_size']
    x = jax.random.uniform(keys[0], (b, s, din), dtype)
    w1 = jax.random.normal(keys[1], (din, dout), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (dout,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (dout, dout), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (dout,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (dout, dout), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (dout,), dtype) * 0.02
    gamma = jnp.ones((dout,), dtype)
    beta = jnp.zeros((dout,), dtype)
    return x, w1, b1, w2, b2, w3, b3, gamma, beta


def workload(x, w1, b1, w2, b2, w3, b3, gamma, beta):
    alpha = CONFIG['celu_alpha']
    pad_size = w1.shape[1] - x.shape[-1]
    padded = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad_size)])
    h = jax.nn.celu(x @ w1 + b1, alpha=alpha)
    g = jax.nn.celu(h @ w2 + b2, alpha=alpha)
    m = g * jnp.tanh(g @ w3 + b3)
    y = padded + m
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    return (y - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta
