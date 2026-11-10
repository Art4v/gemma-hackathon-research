"""[bf16 variant] F.normalize(p=2, dim=-1) on both inputs followed by nn.CosineSimilarity(dim=-1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CosineSimilarity_8255',
    'shape': (8, 128, 128, 128),
    'dim': -1,
    'normalize_eps': 1e-12,
    'cosine_eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_i, k_j = jax.random.split(key)
    i = jax.random.uniform(k_i, CONFIG['shape'], dtype)
    j = jax.random.uniform(k_j, CONFIG['shape'], dtype)
    return i, j


def _l2_norm(x, axis):
    return jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True))


def _normalize(x, axis, eps):
    return x / jnp.maximum(_l2_norm(x, axis), eps)


def workload(i, j):
    axis = CONFIG['dim']
    i = _normalize(i, axis, CONFIG['normalize_eps'])
    j = _normalize(j, axis, CONFIG['normalize_eps'])
    i = _normalize(i, axis, CONFIG['cosine_eps'])
    j = _normalize(j, axis, CONFIG['cosine_eps'])
    return jnp.sum(i * j, axis=axis)
