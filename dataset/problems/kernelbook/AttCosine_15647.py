"""[bf16 variant] AttCosine: cosine similarity over dim 3 of broadcast (query, y) pairs, max-reduce for sim, softmax over keys."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttCosine_15647',
    'batch': 8,
    'n_query': 256,
    'n_key': 256,
    'feat': 128,
    'depth': 128,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    kq, ky = jax.random.split(jax.random.key(0))
    c = CONFIG
    query = jax.random.uniform(kq, (c['batch'], c['n_query'], c['feat'], c['depth']), dtype)
    y = jax.random.uniform(ky, (c['batch'], c['n_key'], c['feat'], c['depth']), dtype)
    return query, y


def workload(query, y):
    eps = CONFIG['eps']
    # Normalize along the cosine axis (C, axis 2 of the 4D inputs); each norm clamped to eps like ATen.
    qn = query / jnp.maximum(jnp.linalg.norm(query, axis=2, keepdims=True), eps)
    yn = y / jnp.maximum(jnp.linalg.norm(y, axis=2, keepdims=True), eps)
    att = jnp.einsum('abcd,aecd->abed', qn, yn)  # (A, B, E, D)
    sim = jnp.max(att, axis=2)[:, None]  # (A, 1, B, D)
    att = jax.nn.softmax(att, axis=2)
    return att, sim
