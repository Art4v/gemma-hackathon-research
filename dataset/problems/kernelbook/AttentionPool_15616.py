"""AttentionPool: Linear -> clamp(-15, 15) -> softmax(pooldim) -> weighted sum of decision / (sum of weights + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionPool_15616',
    'batch': 16,
    'time': 128,
    'freq': 128,
    'inputdim': 128,
    'outputdim': 128,
    'pooldim': 1,
    'clamp_min': -15.0,
    'clamp_max': 15.0,
    'eps': 1e-7,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    logits = jax.random.uniform(k1, (c['batch'], c['time'], c['freq'], c['inputdim']), dtype)
    decision = jax.random.uniform(k2, (c['batch'], c['time'], c['freq'], c['outputdim']), dtype)
    w = jax.random.normal(k3, (c['inputdim'], c['outputdim']), dtype) * 0.02
    b = jax.random.normal(k4, (c['outputdim'],), dtype) * 0.02
    return logits, decision, w, b


def workload(logits, decision, w, b):
    c = CONFIG
    pd = c['pooldim']
    t = jnp.clip(logits @ w + b, c['clamp_min'], c['clamp_max'])
    a = jax.nn.softmax(t, axis=pd)
    return jnp.sum(decision * a, axis=pd) / (jnp.sum(a, axis=pd) + c['eps'])
