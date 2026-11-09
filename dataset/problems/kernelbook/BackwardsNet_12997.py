"""[bf16 variant] Concat -> Linear -> ReLU -> Linear -> soft-label CrossEntropyLoss (mean)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BackwardsNet_12997',
    'batch': 8192,
    'h': 2048,
    'ydim': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    b, h, ydim = CONFIG['batch'], CONFIG['h'], CONFIG['ydim']
    keys = jax.random.split(jax.random.key(0), 7)
    phi_prev = jax.random.uniform(keys[0], (b, h), dtype)
    phi = jax.random.uniform(keys[1], (b, h), dtype)
    atn = jax.random.uniform(keys[2], (b, ydim), dtype)
    w1 = jax.random.normal(keys[3], (2 * h, h), dtype) * 0.02
    b1 = jax.random.normal(keys[4], (h,), dtype) * 0.02
    w2 = jax.random.normal(keys[5], (h, ydim), dtype) * 0.02
    b2 = jax.random.normal(keys[6], (ydim,), dtype) * 0.02
    return phi_prev, phi, atn, w1, b1, w2, b2


def workload(phi_prev, phi, atn, w1, b1, w2, b2):
    x = jnp.concatenate((phi_prev, phi), axis=1)
    x = jax.nn.relu(x @ w1 + b1)
    logits = x @ w2 + b2
    # Float target with the same shape as the logits = probability targets in torch CrossEntropyLoss.
    logp = jax.nn.log_softmax(logits, axis=1)
    return jnp.mean(-jnp.sum(atn * logp, axis=1))
