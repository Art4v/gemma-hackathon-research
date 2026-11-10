"""[bf16 variant] TSA_Fusion_1816: temporal correlation attention (conv, sigmoid) -> fusion conv -> pyramid spatial attention (max/avg pool, conv, bilinear upsample) -> attention-modulated fusion."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TSA_Fusion_1816',
    'batch': 8,
    'nframes': 5,
    'nf': 128,
    'height': 128,
    'width': 128,
    'center': 2,
    'leaky_slope': 0.1,
    'pool_kernel': 3,
    'pool_stride': 2,
    'pool_padding': 1,
    'upsample_scale': 2,
    'att_scale': 2.0,
    'conv_stride': 1,
    'conv_padding': {1: 0, 3: 1},
    'weight_scale': 0.02,
    # (layer name, in_channels, kernel size); out_channels is always nf.
    'conv_layers': (
        ('tAtt_1', 128, 3),
        ('tAtt_2', 128, 3),
        ('fea_fusion', 640, 1),
        ('sAtt_1', 640, 1),
        ('sAtt_2', 256, 1),
        ('sAtt_3', 128, 3),
        ('sAtt_4', 128, 1),
        ('sAtt_5', 128, 3),
        ('sAtt_L1', 128, 1),
        ('sAtt_L2', 256, 3),
        ('sAtt_L3', 128, 3),
        ('sAtt_add_1', 128, 1),
        ('sAtt_add_2', 128, 1),
    ),
}


def create_inputs(dtype=jnp.bfloat16):
    layers = CONFIG['conv_layers']
    nf = CONFIG['nf']
    scale = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(layers))
    shape = (CONFIG['batch'], CONFIG['nframes'], nf, CONFIG['height'], CONFIG['width'])
    inputs = [jax.random.uniform(keys[0], shape, dtype)]
    for i, (_, in_channels, kernel) in enumerate(layers):
        inputs.append(jax.random.normal(keys[1 + 2 * i], (nf, in_channels, kernel, kernel), dtype) * scale)
        inputs.append(jax.random.normal(keys[2 + 2 * i], (nf,), dtype) * scale)
    return tuple(inputs)


def _conv(x, w, b):
    pad = CONFIG['conv_padding'][w.shape[-1]]
    stride = CONFIG['conv_stride']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def _lrelu(x):
    return jax.nn.leaky_relu(x, negative_slope=CONFIG['leaky_slope'])


def _pool_args():
    k = CONFIG['pool_kernel']
    s = CONFIG['pool_stride']
    p = CONFIG['pool_padding']
    return (1, k, k, 1), (1, s, s, 1), [(0, 0), (p, p), (p, p), (0, 0)]


def _maxpool(x):
    window, strides, padding = _pool_args()
    return lax.reduce_window(x, -jnp.inf, lax.max, window, strides, padding)


def _avgpool(x):
    window, strides, padding = _pool_args()
    k = CONFIG['pool_kernel']
    return lax.reduce_window(x, 0.0, lax.add, window, strides, padding) / (k * k)


def _resize_axis(x, axis, scale):
    n = x.shape[axis]
    pos = (jnp.arange(n * scale, dtype=x.dtype) + 0.5) / scale - 0.5
    lo = jnp.floor(pos)
    frac = (pos - lo).reshape([-1 if a == axis else 1 for a in range(x.ndim)])
    i0 = jnp.clip(lo.astype(jnp.int32), 0, n - 1)
    i1 = jnp.clip(lo.astype(jnp.int32) + 1, 0, n - 1)
    return jnp.take(x, i0, axis=axis) * (1 - frac) + jnp.take(x, i1, axis=axis) * frac


def _upsample(x):
    scale = CONFIG['upsample_scale']
    return _resize_axis(_resize_axis(x, 1, scale), 2, scale)


def workload(
    aligned_fea,
    tAtt_1_w, tAtt_1_b,
    tAtt_2_w, tAtt_2_b,
    fea_fusion_w, fea_fusion_b,
    sAtt_1_w, sAtt_1_b,
    sAtt_2_w, sAtt_2_b,
    sAtt_3_w, sAtt_3_b,
    sAtt_4_w, sAtt_4_b,
    sAtt_5_w, sAtt_5_b,
    sAtt_L1_w, sAtt_L1_b,
    sAtt_L2_w, sAtt_L2_b,
    sAtt_L3_w, sAtt_L3_b,
    sAtt_add_1_w, sAtt_add_1_b,
    sAtt_add_2_w, sAtt_add_2_b,
):
    B, N, C, H, W = aligned_fea.shape
    x = jnp.transpose(aligned_fea, (0, 1, 3, 4, 2))  # B, N, H, W, C

    emb_ref = _conv(x[:, CONFIG['center']], tAtt_2_w, tAtt_2_b)
    emb = _conv(x.reshape((B * N, H, W, C)), tAtt_1_w, tAtt_1_b).reshape((B, N, H, W, -1))
    cor_prob = jax.nn.sigmoid(jnp.sum(emb * emb_ref[:, None], axis=-1))  # B, N, H, W

    weighted = x * cor_prob[..., None]
    fused = jnp.transpose(weighted, (0, 2, 3, 1, 4)).reshape((B, H, W, N * C))

    fea = _lrelu(_conv(fused, fea_fusion_w, fea_fusion_b))
    att = _lrelu(_conv(fused, sAtt_1_w, sAtt_1_b))

    att_pooled = jnp.concatenate([_maxpool(att), _avgpool(att)], axis=-1)
    att = _lrelu(_conv(att_pooled, sAtt_2_w, sAtt_2_b))

    att_L = _lrelu(_conv(att, sAtt_L1_w, sAtt_L1_b))
    att_L_pooled = jnp.concatenate([_maxpool(att_L), _avgpool(att_L)], axis=-1)
    att_L = _lrelu(_conv(att_L_pooled, sAtt_L2_w, sAtt_L2_b))
    att_L = _lrelu(_conv(att_L, sAtt_L3_w, sAtt_L3_b))
    att_L = _upsample(att_L)

    att = _lrelu(_conv(att, sAtt_3_w, sAtt_3_b))
    att = att + att_L
    att = _lrelu(_conv(att, sAtt_4_w, sAtt_4_b))
    att = _upsample(att)
    att = _conv(att, sAtt_5_w, sAtt_5_b)

    att_add = _conv(_lrelu(_conv(att, sAtt_add_1_w, sAtt_add_1_b)), sAtt_add_2_w, sAtt_add_2_b)
    att = jax.nn.sigmoid(att)

    out = fea * att * CONFIG['att_scale'] + att_add
    return jnp.transpose(out, (0, 3, 1, 2))
