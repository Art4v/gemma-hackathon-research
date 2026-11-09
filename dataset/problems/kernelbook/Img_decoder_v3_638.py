"""[bf16 variant] Img_decoder_v3: 4x [bilinear up2 -> reflect-pad conv3x3 -> LeakyReLU -> InstanceNorm (x2) + 1x1 shortcut] -> softmax-weighted conv head sum."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Img_decoder_v3_638',
    'batch': 8,
    'in_channels': 512,
    'height': 64,
    'width': 64,
    # (in_ch, out_ch, upsample) per residual block
    'blocks': ((512, 256, True), (256, 128, True), (128, 64, True), (64, 64, False)),
    'head_channels': 5,
    'scale_factor': 2,
    'negative_slope': 0.2,
    'eps': 1e-5,
}

_DN = ('NHWC', 'HWIO', 'NHWC')


def create_inputs(dtype=jnp.bfloat16):
    blocks = CONFIG['blocks']
    hc = CONFIG['head_channels']
    keys = jax.random.split(jax.random.key(0), 1 + 5 * len(blocks) + 4)
    feat = jax.random.uniform(
        keys[0], (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    params = []
    k = 1
    for cin, cout, _ in blocks:
        shapes = [(3, 3, cin, cout), (cout,), (3, 3, cout, cout), (cout,), (1, 1, cin, cout)]
        for s in shapes:
            params.append(jax.random.normal(keys[k], s, dtype) * 0.02)
            k += 1
    c_last = blocks[-1][1]
    for s in [(3, 3, c_last, hc), (hc,), (3, 3, c_last, hc), (hc,)]:
        params.append(jax.random.normal(keys[k], s, dtype) * 0.02)
        k += 1
    return (feat, *params)


def _conv(x, w, padding):
    return lax.conv_general_dilated(x, w, (1, 1), padding, dimension_numbers=_DN,
                                    precision=lax.Precision.HIGHEST)


def _leaky(x):
    return jnp.where(x >= 0, x, x * CONFIG['negative_slope'])


def _instance_norm(x):
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps'])


def _reflect_conv(x, w, b):
    x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='reflect')
    return _conv(x, w, 'VALID') + b


def _res_block(x, w1, b1, w2, b2, ws, upsample):
    if upsample:
        n, h, wd, c = x.shape
        s = CONFIG['scale_factor']
        x = jax.image.resize(x, (n, h * s, wd * s, c), method='bilinear')
    x_s = _conv(x, ws, 'VALID')
    x = _instance_norm(_leaky(_reflect_conv(x, w1, b1)))
    x = _instance_norm(_leaky(_reflect_conv(x, w2, b2)))
    return x_s + x


def workload(feat, *params):
    blocks = CONFIG['blocks']
    h = jnp.transpose(feat, (0, 2, 3, 1))
    for i, (_, _, upsample) in enumerate(blocks):
        h = _res_block(h, *params[5 * i:5 * i + 5], upsample)
    map_w, map_b, conf_w, conf_b = params[5 * len(blocks):]
    pad = ((1, 1), (1, 1))
    score = jax.nn.softmax(_conv(h, conf_w, pad) + conf_b, axis=-1)
    out = jnp.sum((_conv(h, map_w, pad) + map_b) * score, axis=-1)
    return out[:, None, :, :]
