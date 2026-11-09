"""MLP: Linear(no bias) -> LayerNorm -> SiLU -> Linear(bias)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_17260',
    'batch': 8,
    'd1': 128,
    'd2': 128,
    'ni': 512,
    'nhidden': 512,
    'no': 512,
    'nlayers': 1,  # ponytail: workload implements exactly one hidden layer
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, k0, k1, k2 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (c['batch'], c['d1'], c['d2'], c['ni']), dtype)
    w0 = jax.random.normal(k0, (c['ni'], c['nhidden']), dtype) * 0.02
    gamma = jnp.ones((c['nhidden'],), dtype)
    beta = jnp.zeros((c['nhidden'],), dtype)
    w_out = jax.random.normal(k1, (c['nhidden'], c['no']), dtype) * 0.02
    b_out = jax.random.normal(k2, (c['no'],), dtype) * 0.02
    return x, w0, gamma, beta, w_out, b_out


def workload(x, w0, gamma, beta, w_out, b_out):
    h = x @ w0
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(h - mean), axis=-1, keepdims=True)
    h = (h - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta
    h = h * jax.nn.sigmoid(h)
    return h @ w_out + b_out
