"""[bf16 variant] CXLoss: center by target, channel L2 normalize, per-sample 1x1 patch conv, relative distance, contextual similarity softmax, max over positions, -log mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CXLoss_15263',
    'batch': 8,
    'channels': 256,
    'height': 64,
    'width': 64,
    'sigma': 0.1,
    'b': 1.0,
    'norm_eps': 1e-6,
    'relative_eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    key_t, key_i = jax.random.split(key)
    feature_t = jax.random.uniform(key_t, shape, dtype)
    feature_i = jax.random.uniform(key_i, shape, dtype)
    return feature_t, feature_i


def _l2_normalize_channelwise(features):
    norms = jnp.sqrt(jnp.sum(features * features, axis=1, keepdims=True))
    return features / (norms + CONFIG['norm_eps'])


def _patch_conv(sample):
    feature_t_i, feature_i_i = sample
    c, h, w = feature_t_i.shape
    kernel = jnp.reshape(feature_t_i, (1, 1, c, h * w))
    lhs = jnp.transpose(feature_i_i, (1, 2, 0))[None, ...]
    out = lax.conv_general_dilated(
        lhs,
        kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(out, (0, 3, 1, 2))[0]


def workload(feature_t, feature_i):
    mean_t = jnp.mean(feature_t, axis=0, keepdims=True)
    mean_t = jnp.mean(mean_t, axis=2, keepdims=True)
    mean_t = jnp.mean(mean_t, axis=3, keepdims=True)
    feature_i = _l2_normalize_channelwise(feature_i - mean_t)
    feature_t = _l2_normalize_channelwise(feature_t - mean_t)

    dist = lax.map(_patch_conv, (feature_t, feature_i))

    raw_dist = jnp.maximum((1.0 - dist) / 2.0, 0.0)
    div = jnp.min(raw_dist, axis=1, keepdims=True)
    relative_dist = raw_dist / (div + CONFIG['relative_eps'])

    weights = jnp.exp((CONFIG['b'] - relative_dist) / CONFIG['sigma'])
    weights_sum = jnp.sum(weights, axis=1, keepdims=True)
    cx = weights / weights_sum

    cx = jnp.max(jnp.max(cx, axis=3), axis=2)
    cx = jnp.mean(cx, axis=1)
    cx = -jnp.log(cx)
    return jnp.mean(cx)
