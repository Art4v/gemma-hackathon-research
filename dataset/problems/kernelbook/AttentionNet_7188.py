"""[bf16 variant] AttentionNet: 2-level U-Net (conv3x3+ReLU x2, maxpool, bilinear upsample, concat skip) + 1x1 conv."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AttentionNet_7188',
    'batch': 16,
    'in_channels': 3,
    'out_channels': 1,
    'height': 512,
    'width': 512,
    'filters': (64, 128, 256),
    'kernel_size': 3,
    'pool': 2,
    'upsample_scale': 2,
}


def _layer_specs():
    f0, f1, f2 = CONFIG['filters']
    k = CONFIG['kernel_size']
    return [
        (CONFIG['in_channels'], f0, k), (f0, f0, k),   # down1
        (f0, f1, k), (f1, f1, k),                      # down2
        (f1, f2, k), (f2, f2, k),                      # center
        (f2 + f1, f1, k), (f1, f1, k),                 # up2
        (f1 + f0, f0, k), (f0, f0, k),                 # up1
        (f0, CONFIG['out_channels'], 1),               # final 1x1
    ]


def create_inputs(dtype=jnp.bfloat16):
    specs = _layer_specs()
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(specs))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i, (cin, cout, k) in enumerate(specs):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, cin, cout), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def _conv(x, w, b):
    y = lax.conv_general_dilated(
        x, w, (1, 1), 'SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def _conv_relu2(x, p):
    x = jax.nn.relu(_conv(x, p[0], p[1]))
    return jax.nn.relu(_conv(x, p[2], p[3]))


def _maxpool(x):
    p = CONFIG['pool']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def _upsample(x):
    s = CONFIG['upsample_scale']
    n, h, w, c = x.shape
    # half-pixel bilinear == PyTorch align_corners=False for upsampling
    return jax.image.resize(x, (n, h * s, w * s, c), method='bilinear')


def _decoder(skip, x, p):
    return _conv_relu2(jnp.concatenate([skip, _upsample(x)], axis=-1), p)


def workload(x, *params):
    x = jnp.transpose(x, (0, 2, 3, 1))
    bef1 = _conv_relu2(x, params[0:4])
    x = _maxpool(bef1)
    bef2 = _conv_relu2(x, params[4:8])
    x = _maxpool(bef2)
    x = _conv_relu2(x, params[8:12])
    x = _decoder(bef2, x, params[12:16])
    x = _decoder(bef1, x, params[16:20])
    x = _conv(x, params[20], params[21])
    return jnp.transpose(x, (0, 3, 1, 2))
