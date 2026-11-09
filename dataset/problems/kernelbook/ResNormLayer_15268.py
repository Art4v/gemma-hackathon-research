"""ResNormLayer: Linear -> ReLU -> LayerNorm -> Linear -> ReLU -> LayerNorm -> residual add."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResNormLayer_15268',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'linear_size': 1024,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    b, d1, d2, l = CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['linear_size']
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (b, d1, d2, l), dtype)
    w1 = jax.random.normal(kw1, (l, l), dtype) * 0.02
    b1 = jax.random.normal(kb1, (l,), dtype) * 0.02
    w2 = jax.random.normal(kw2, (l, l), dtype) * 0.02
    b2 = jax.random.normal(kb2, (l,), dtype) * 0.02
    g1 = jnp.ones((l,), dtype)
    beta1 = jnp.zeros((l,), dtype)
    g2 = jnp.ones((l,), dtype)
    beta2 = jnp.zeros((l,), dtype)
    return x, w1, b1, g1, beta1, w2, b2, g2, beta2


def _layer_norm(y, gamma, beta):
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    return (y - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, w1, b1, g1, beta1, w2, b2, g2, beta2):
    y = jax.nn.relu(x @ w1 + b1)
    y = _layer_norm(y, g1, beta1)
    y = jax.nn.relu(y @ w2 + b2)
    y = _layer_norm(y, g2, beta2)
    return x + y
