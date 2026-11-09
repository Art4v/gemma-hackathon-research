"""Multinomial KL divergence: sum(q * (log q - log p)) / (dim1 * dim0)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultinomialKLDivergenceLoss_7714',
    'shape': (16, 128, 128, 128),
    'minval': 1e-6,  # keeps log() finite; torch.rand is [0, 1)
    'maxval': 1.0,
}


def create_inputs(dtype=jnp.float32):
    kp, kq = jax.random.split(jax.random.key(0))
    shape = CONFIG['shape']
    p_proba = jax.random.uniform(kp, shape, dtype, CONFIG['minval'], CONFIG['maxval'])
    q_proba = jax.random.uniform(kq, shape, dtype, CONFIG['minval'], CONFIG['maxval'])
    return p_proba, q_proba


def workload(p_proba, q_proba):
    loss = jnp.sum(q_proba * (jnp.log(q_proba) - jnp.log(p_proba)))
    return loss / (p_proba.shape[1] * p_proba.shape[0])
