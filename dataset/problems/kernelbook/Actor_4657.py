"""[bf16 variant] Actor MLP: Linear -> ELU -> LayerNorm -> Linear -> ELU -> LayerNorm -> Linear -> sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_4657',
    'batch': 65536,
    'state_dim': 512,
    'l1': 400,
    'l2': 300,
    'action_dim': 128,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (c['batch'], c['state_dim']), dtype)
    w1 = jax.random.normal(ks[1], (c['state_dim'], c['l1']), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (c['l1'],), dtype) * 0.02
    g1 = jnp.ones((c['l1'],), dtype)
    be1 = jnp.zeros((c['l1'],), dtype)
    w2 = jax.random.normal(ks[3], (c['l1'], c['l2']), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c['l2'],), dtype) * 0.02
    g2 = jnp.ones((c['l2'],), dtype)
    be2 = jnp.zeros((c['l2'],), dtype)
    w3 = jax.random.normal(ks[5], (c['l2'], c['action_dim']), dtype) * 0.02
    b3 = jax.random.normal(ks[6], (c['action_dim'],), dtype) * 0.02
    return (x, w1, b1, g1, be1, w2, b2, g2, be2, w3, b3)


def _layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, w1, b1, g1, be1, w2, b2, g2, be2, w3, b3):
    out = jax.nn.elu(x @ w1 + b1)
    out = _layer_norm(out, g1, be1)
    out = jax.nn.elu(out @ w2 + b2)
    out = _layer_norm(out, g2, be2)
    return jax.nn.sigmoid(out @ w3 + b3)
