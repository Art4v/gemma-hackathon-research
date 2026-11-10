"""Weighted binary cross-entropy: class-balance weights from label counts, then mean BCE with clamped logs."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Weighed_Bce_Loss_16513',
    'batch': 16,
    'height': 1024,
    'width': 2048,
    'label_p': 0.5,
    'log_clamp': -100.0,
}


def create_inputs(dtype=jnp.float32):
    kx, kl = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    # Binary labels so the p / 1-p weighting is exercised; rand labels give all-zero weights.
    label = jax.random.bernoulli(kl, CONFIG['label_p'], shape).astype(dtype)
    return x, label


def workload(x, label):
    is_t = label == 1
    is_f = label == 0
    n_t = jnp.sum(is_t.astype(jnp.float32))
    n_f = jnp.sum(is_f.astype(jnp.float32))
    p = n_t / (n_t + n_f)
    # Elements with label neither 0 nor 1 keep the zeros_like weight, as in the torch module.
    w = jnp.where(is_t, p, jnp.where(is_f, 1.0 - p, 0.0)).astype(x.dtype)
    clamp = CONFIG['log_clamp']
    log_x = jnp.maximum(jnp.log(x), clamp)
    log_1mx = jnp.maximum(jnp.log(1.0 - x), clamp)
    bce = -(label * log_x + (1.0 - label) * log_1mx)
    return jnp.mean(w * bce)
