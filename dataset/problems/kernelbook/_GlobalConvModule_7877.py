"""Global Convolution Module: Dropout2d(identity) -> [Conv(kh,1) -> Conv(1,kw)] + [Conv(1,kw) -> Conv(kh,1)], with biases."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '_GlobalConvModule_7877',
    'batch': 8,
    'in_dim': 128,
    'out_dim': 128,
    'height': 256,
    'width': 256,
    'kernel_h': 4,
    'kernel_w': 4,
    'dropout_p': 0.1,  # Dropout2d is identity in inference mode
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    Ci = CONFIG['in_dim']
    Co = CONFIG['out_dim']
    H = CONFIG['height']
    W = CONFIG['width']
    kh = CONFIG['kernel_h']
    kw = CONFIG['kernel_w']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (B, Ci, H, W), dtype)
    # Conv weights in HWIO layout.
    w_l1 = jax.random.normal(keys[1], (kh, 1, Ci, Co), dtype) * 0.02
    b_l1 = jax.random.normal(keys[2], (Co,), dtype) * 0.02
    w_l2 = jax.random.normal(keys[3], (1, kw, Co, Co), dtype) * 0.02
    b_l2 = jax.random.normal(keys[4], (Co,), dtype) * 0.02
    w_r1 = jax.random.normal(keys[5], (1, kw, Ci, Co), dtype) * 0.02
    b_r1 = jax.random.normal(keys[6], (Co,), dtype) * 0.02
    w_r2 = jax.random.normal(keys[7], (kh, 1, Co, Co), dtype) * 0.02
    b_r2 = jax.random.normal(keys[8], (Co,), dtype) * 0.02
    return (x, w_l1, b_l1, w_l2, b_l2, w_r1, b_r1, w_r2, b_r2)


def _conv(x_nhwc, w_hwio, b, pad_h, pad_w):
    y = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b.reshape(1, 1, 1, -1)


def workload(x, w_l1, b_l1, w_l2, b_l2, w_r1, b_r1, w_r2, b_r2):
    # PyTorch padding: (k - 1) // 2 on both sides (even k shrinks the axis by 1).
    kh = w_l1.shape[0]
    kw = w_l2.shape[1]
    pad0 = (kh - 1) // 2
    pad1 = (kw - 1) // 2
    # Dropout2d -> identity (inference).
    xh = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    x_l = _conv(xh, w_l1, b_l1, pad0, 0)
    x_l = _conv(x_l, w_l2, b_l2, 0, pad1)
    x_r = _conv(xh, w_r1, b_r1, 0, pad1)
    x_r = _conv(x_r, w_r2, b_r2, pad0, 0)
    y = x_l + x_r
    return jnp.transpose(y, (0, 3, 1, 2))  # NHWC -> NCHW
