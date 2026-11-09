"""[bf16 variant] Circle loss: x @ x.T similarity, label masks, clamped weights, two logsumexps, softplus, mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'circle_loss',
    'batch': 8192,
    'dim': 2048,
    'num_classes': 128,
    'm': 4.0,
    'gamma': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kt = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['dim']), dtype)
    target = jax.random.randint(kt, (CONFIG['batch'],), 0, CONFIG['num_classes'], jnp.int32)
    return x, target


def _diagonal_mask(shape):
    # torch fill_diagonal_: positions where every index is equal (works for any rank).
    first = lax.broadcasted_iota(jnp.int32, shape, 0)
    mask = jnp.ones(shape, dtype=bool)
    for d in range(1, len(shape)):
        mask = mask & (lax.broadcasted_iota(jnp.int32, shape, d) == first)
    return mask


def workload(x, target):
    m = CONFIG['m']
    gamma = CONFIG['gamma']
    dp = 1.0 - m
    dn = m
    sim = jnp.matmul(x, x.T, precision=lax.Precision.HIGHEST)
    label = jnp.expand_dims(target, 1) == jnp.expand_dims(target, 0)
    negative = jnp.logical_not(label)
    positive = label & jnp.logical_not(_diagonal_mask(label.shape))
    zeros = jnp.zeros_like(sim)
    sp = jnp.where(positive, sim, zeros)
    sn = jnp.where(negative, sim, zeros)
    ap = jnp.maximum(1.0 + m - lax.stop_gradient(sp), 0.0)
    an = jnp.maximum(lax.stop_gradient(sn) + m, 0.0)
    logit_p = jnp.where(positive, -gamma * ap * (sp - dp), 0.0)
    logit_n = jnp.where(negative, gamma * an * (sn - dn), 0.0)
    lse = jax.nn.logsumexp(logit_p, axis=1) + jax.nn.logsumexp(logit_n, axis=1)
    return jnp.mean(jax.nn.softplus(lse))
