"""Negative binomial NLL: chunk(preds) -> log/lgamma terms -> clamp(-300,300) -> negate -> mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LossLoglikelihoodNb_3760',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'clamp_min': -300.0,
    'clamp_max': 300.0,
    'preds_minval': 1e-3,
    'average': True,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    # minval keeps log(loc)/log(scale) finite; torch.rand could hit exactly 0.
    preds = jax.random.uniform(k1, (b, 2 * c, h, w), dtype, minval=CONFIG['preds_minval'], maxval=1.0)
    target = jax.random.uniform(k2, (b, c, h, w), dtype)
    return preds, target


def workload(preds, target):
    x = target
    loc, scale = jnp.split(preds, 2, axis=1)
    eta_loc = jnp.log(loc)
    eta_scale = jnp.log(scale)
    log_r_plus_mu = jnp.log(scale + loc)
    ll = lax.lgamma(scale + x)
    ll = ll - lax.lgamma(x + jnp.ones_like(x))
    ll = ll - lax.lgamma(scale)
    ll = ll + x * (eta_loc - log_r_plus_mu) + scale * (eta_scale - log_r_plus_mu)
    ll = jnp.clip(ll, CONFIG['clamp_min'], CONFIG['clamp_max'])
    neg_ll = -ll
    if CONFIG['average']:
        return jnp.mean(neg_ll)
    return neg_ll.sum(axis=1).sum(axis=1)
