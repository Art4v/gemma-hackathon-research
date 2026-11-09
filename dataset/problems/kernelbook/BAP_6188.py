"""BAP: bilinear attention pooling, global average pool of features * each attention map, stacked to (B, M, C)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BAP_6188',
    'batch': 16,
    'channels': 512,
    'num_attentions': 32,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.float32):
    k_feat, k_att = jax.random.split(jax.random.key(0))
    b, c, m = CONFIG['batch'], CONFIG['channels'], CONFIG['num_attentions']
    h, w = CONFIG['height'], CONFIG['width']
    features = jax.random.uniform(k_feat, (b, c, h, w), dtype)
    attentions = jax.random.uniform(k_att, (b, m, h, w), dtype)
    return features, attentions


def workload(features, attentions):
    h, w = features.shape[2], features.shape[3]
    # ponytail: the einsum is the same as mean_hw(features * att_i) for every i, without the 5D product
    pooled = jnp.einsum('bchw,bmhw->bmc', features, attentions,
                        precision=lax.Precision.HIGHEST)
    return pooled / (h * w)
