"""RegressionModel: 4x (Conv3x3 + ReLU) -> Conv3x3 to num_anchors*8 -> permute NHWC -> reshape (N, -1, 8)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'RegressionModel_2877',
    'batch': 8,
    'num_features_in': 256,
    'height': 128,
    'width': 128,
    'feature_size': 256,
    'num_anchors': 1,
    'box_dim': 8,
    'kernel_size': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(
        keys[0], (c['batch'], c['num_features_in'], c['height'], c['width']), dtype)
    chans = [c['num_features_in']] + [c['feature_size']] * 4 + [c['num_anchors'] * c['box_dim']]
    params = []
    for i in range(5):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, chans[i], chans[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def _conv(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    out = jnp.transpose(x, (0, 2, 3, 1))
    for w, b in ((w1, b1), (w2, b2), (w3, b3), (w4, b4)):
        out = jax.nn.relu(_conv(out, w, b))
    out = _conv(out, w5, b5)
    # NHWC already matches PyTorch's permute(0, 2, 3, 1).
    box_dim = CONFIG['box_dim']
    return out.reshape(out.shape[0], -1, box_dim)
