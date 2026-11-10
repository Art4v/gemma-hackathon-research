"""[bf16 variant] RetinaNet ClassificationModel: 4x (Conv3x3 + ReLU) -> Conv3x3 -> Sigmoid -> reshape (B, H*W*A, K)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ClassificationModel_422',
    'batch': 8,
    'num_features_in': 256,
    'height': 128,
    'width': 128,
    'feature_size': 256,
    'num_anchors': 9,
    'num_classes': 80,
    'kernel_size': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 11)
    k = c['kernel_size']
    x = jax.random.uniform(
        keys[0], (c['batch'], c['num_features_in'], c['height'], c['width']), dtype)
    chans = [c['num_features_in']] + [c['feature_size']] * 4 + [c['num_anchors'] * c['num_classes']]
    params = []
    for i in range(5):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, chans[i], chans[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def _conv(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    out = jnp.transpose(x, (0, 2, 3, 1))
    for w, b in ((w1, b1), (w2, b2), (w3, b3), (w4, b4)):
        out = jax.nn.relu(_conv(out, w, b))
    out = jax.nn.sigmoid(_conv(out, w5, b5))
    # NHWC already matches torch's permute(0, 2, 3, 1); flatten H, W and anchors.
    return out.reshape(out.shape[0], -1, CONFIG['num_classes'])
