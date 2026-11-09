"""[bf16 variant] InceptionA: 1x1 conv, 1x1->5x5 conv, 1x1->3x3->3x3 conv, avgpool3x3->1x1 conv, channel concat."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'InceptionA_5869',
    'batch': 16,
    'in_channels': 128,
    'height': 128,
    'width': 128,
    'c1x1': 16,
    'c5x5': 24,
    'c3x3': 24,
    'cpool': 24,
    'k5': 5,
    'k3': 3,
    'pool_k': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(ks[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    shapes = [
        (c['c1x1'], c['in_channels'], 1, 1),
        (c['c5x5'], c['c1x1'], c['k5'], c['k5']),
        (c['c3x3'], c['c1x1'], c['k3'], c['k3']),
        (c['c3x3'], c['c3x3'], c['k3'], c['k3']),
        (c['cpool'], c['in_channels'], 1, 1),
    ]
    params = []
    for i, s in enumerate(shapes):
        params.append(jax.random.normal(ks[1 + 2 * i], s, dtype) * 0.02)
        params.append(jax.random.normal(ks[2 + 2 * i], (s[0],), dtype) * 0.02)
    return (x, *params)


def _conv(x, w, b):
    # x NHWC, w OIHW (PyTorch layout), 'same' padding for odd kernels, stride 1.
    kh, kw = w.shape[2], w.shape[3]
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1),
        padding=((kh // 2, kh // 2), (kw // 2, kw // 2)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def _avg_pool(x, k):
    # count_include_pad=True (PyTorch default): divide by full window area.
    p = k // 2
    s = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add,
                          (1, k, k, 1), (1, 1, 1, 1),
                          ((0, 0), (p, p), (p, p), (0, 0)))
    return s / (k * k)


def workload(x, w1, b1, w5, b5, w3_2, b3_2, w3_3, b3_3, wpool, bpool):
    xh = jnp.transpose(x, (0, 2, 3, 1))
    b1x1 = _conv(xh, w1, b1)
    b5x5 = _conv(b1x1, w5, b5)
    b3x3 = _conv(_conv(b1x1, w3_2, b3_2), w3_3, b3_3)
    bpl = _conv(_avg_pool(xh, CONFIG['pool_k']), wpool, bpool)
    out = jnp.concatenate([b1x1, b5x5, b3x3, bpl], axis=-1)
    return jnp.transpose(out, (0, 3, 1, 2))
