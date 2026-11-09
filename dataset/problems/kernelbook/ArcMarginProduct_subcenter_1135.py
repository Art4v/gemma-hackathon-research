"""[bf16 variant] ArcMarginProduct_subcenter: normalize(features, dim=1) @ normalize(weight).T -> view(-1, out, k) -> max over k."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ArcMarginProduct_subcenter_1135',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'in_features': 512,
    'out_features': 512,
    'k': 3,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w = jax.random.split(jax.random.key(0))
    features = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['in_features']),
        dtype,
    )
    weight = jax.random.normal(
        k_w, (CONFIG['in_features'], CONFIG['out_features'] * CONFIG['k']), dtype
    ) * 0.02
    return features, weight


def _l2_normalize(x, axis):
    norm = jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True))
    return x / jnp.maximum(norm, CONFIG['eps'])


def workload(features, weight):
    k = CONFIG['k']
    out_features = weight.shape[1] // k
    # F.normalize defaults to dim=1 (channel axis of the 4D input), not the contracted last axis.
    x = _l2_normalize(features, axis=1)
    w = _l2_normalize(weight, axis=0)
    cosine_all = jnp.matmul(x, w, precision=lax.Precision.HIGHEST)
    cosine_all = cosine_all.reshape(-1, out_features, k)
    return jnp.max(cosine_all, axis=2)
