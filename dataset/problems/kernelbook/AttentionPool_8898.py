"""[bf16 variant] AttentionPool: Linear -> Softmax(dim=1) -> weighted sum over dim 1 / (sum of weights + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionPool_8898',
    'batch': 8,
    'pool_len': 256,
    'frames': 64,
    'inputdim': 256,
    'outputdim': 128,
    'pooldim': 1,
    'eps': 1e-07,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    lead = (c['batch'], c['pool_len'], c['frames'])
    logits = jax.random.uniform(k1, lead + (c['inputdim'],), dtype)
    decision = jax.random.uniform(k2, lead + (c['outputdim'],), dtype)
    w = jax.random.normal(k3, (c['inputdim'], c['outputdim']), dtype) * 0.02
    b = jax.random.normal(k4, (c['outputdim'],), dtype) * 0.02
    return logits, decision, w, b


def workload(logits, decision, w, b):
    axis = CONFIG['pooldim']
    weights = jax.nn.softmax(logits @ w + b, axis=axis)
    return (decision * weights).sum(axis) / (weights.sum(axis) + CONFIG['eps'])
