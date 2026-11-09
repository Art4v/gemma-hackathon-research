"""[bf16 variant] L1Part (OpenPose-style): two stages of 5x [3x(conv3x3 + PReLU) -> concat], then conv1x1 + PReLU + conv1x1, with channel concats between stages."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'L1Part_1682',
    'batch': 8,
    'height': 512,
    'width': 512,
    'feat_channels': 1,
    'l2_channels': 3,
    'stage_out_channels': 4,
    'stage_per_sub': (96, 128),
    'stage_mid': (256, 512),
    'num_concats': 5,
    'subs_per_concat': 3,
    'kernel_size': 3,
    'prelu_init': 0.25,
}


def _param_specs():
    """(kind, shape) for every parameter, in torch model.parameters() order."""
    k = CONFIG['kernel_size']
    subs = CONFIG['subs_per_concat']
    out = CONFIG['stage_out_channels']
    in_ch = CONFIG['feat_channels'] + CONFIG['l2_channels']
    specs = []
    for s, (per, mid) in enumerate(zip(CONFIG['stage_per_sub'], CONFIG['stage_mid'])):
        c = in_ch if s == 0 else in_ch + out
        for _ in range(CONFIG['num_concats']):
            for _ in range(subs):
                specs += [('w', (k, k, c, per)), ('b', (per,)), ('a', (per,))]
                c = per
            c = subs * per
        specs += [('w', (1, 1, c, mid)), ('b', (mid,)), ('a', (mid,)),
                  ('w', (1, 1, mid, out)), ('b', (out,))]
    return specs


def create_inputs(dtype=jnp.bfloat16):
    specs = _param_specs()
    keys = jax.random.split(jax.random.key(0), 2 + len(specs))
    b, h, w = CONFIG['batch'], CONFIG['height'], CONFIG['width']
    features = jax.random.uniform(keys[0], (b, CONFIG['feat_channels'], h, w), dtype)
    l2_out = jax.random.uniform(keys[1], (b, CONFIG['l2_channels'], h, w), dtype)
    params = []
    for key, (kind, shape) in zip(keys[2:], specs):
        if kind == 'a':
            params.append(jnp.full(shape, CONFIG['prelu_init'], dtype))
        else:
            params.append(jax.random.normal(key, shape, dtype) * 0.02)
    return (features, l2_out, *params)


def _conv(x, w, b):
    pad = w.shape[0] // 2
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def _prelu(x, a):
    return jnp.where(x >= 0, x, a * x)


def _stage(x, it):
    for _ in range(CONFIG['num_concats']):
        outs = []
        for _ in range(CONFIG['subs_per_concat']):
            w, b, a = next(it), next(it), next(it)
            x = _prelu(_conv(x, w, b), a)
            outs.append(x)
        x = jnp.concatenate(outs, axis=-1)
        # inputs to the next concatLayer are the concatenated sub outputs
    w6, b6, a6 = next(it), next(it), next(it)
    x = _prelu(_conv(x, w6, b6), a6)
    w7, b7 = next(it), next(it)
    return _conv(x, w7, b7)


def workload(features, l2_out, *params):
    f = jnp.transpose(features, (0, 2, 3, 1))
    l2 = jnp.transpose(l2_out, (0, 2, 3, 1))
    it = iter(params)
    x = _stage(jnp.concatenate([f, l2], axis=-1), it)
    x = _stage(jnp.concatenate([f, x, l2], axis=-1), it)
    return jnp.transpose(x, (0, 3, 1, 2))
