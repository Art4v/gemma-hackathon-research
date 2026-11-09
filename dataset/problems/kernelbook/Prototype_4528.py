"""Prototype layer: flatten to (-1, latent), squared Euclidean cdist to prototypes."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Prototype_4528',
    'input_shape': (16, 64, 128, 128),
    'prototype_num': 512,
    'latent_size': 128,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_p = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_x, CONFIG['input_shape'], dtype)
    prototypes = jax.random.uniform(
        k_p, (CONFIG['prototype_num'], CONFIG['latent_size']), dtype)
    return x, prototypes


def workload(x, prototypes):
    # latent size comes from the parameter shape so any input size works
    latent = prototypes.shape[1]
    xf = x.reshape(-1, latent)
    x2 = jnp.sum(xf * xf, axis=1, keepdims=True)
    y2 = jnp.sum(prototypes * prototypes, axis=1)[None, :]
    xy = jnp.matmul(xf, prototypes.T, precision=lax.Precision.HIGHEST)
    d2 = jnp.maximum(x2 + y2 - 2.0 * xy, 0.0)
    # cdist returns sqrt, the module squares it
    dis_2 = jnp.sqrt(d2) ** 2
    return dis_2, prototypes
