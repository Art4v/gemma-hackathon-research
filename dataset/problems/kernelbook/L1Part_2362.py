"""OpenPose L1Part: concat -> 2 stages of (5 x dense 3x(conv3x3+PReLU) concat blocks, conv1x1+PReLU, conv1x1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'L1Part_2362',
    'batch': 8,
    'height': 128,
    'width': 128,
    'features_channels': 128,
    'l2out_channels': 128,
    'stage_out_channels': 128,
    'per_sub_channels': (96, 128),
    'mid_channels': (256, 512),
    'n_concat': 5,
    'n_sub': 3,
    'prelu_init': 0.25,
}


def _stage_shapes(in_ch, per_sub, mid, out_ch):
    """Shapes of one stage's params, in workload order."""
    shapes = []
    c = in_ch
    for _ in range(CONFIG['n_concat']):
        for _ in range(CONFIG['n_sub']):
            shapes.append(('w', (3, 3, c, per_sub)))
            shapes.append(('b', (per_sub,)))
            shapes.append(('a', (per_sub,)))
            c = per_sub
        c = CONFIG['n_sub'] * per_sub
    shapes += [('w', (1, 1, c, mid)), ('b', (mid,)), ('a', (mid,)),
               ('w', (1, 1, mid, out_ch)), ('b', (out_ch,))]
    return shapes


def create_inputs(dtype=jnp.float32):
    cf, cl, so = CONFIG['features_channels'], CONFIG['l2out_channels'], CONFIG['stage_out_channels']
    b, h, w = CONFIG['batch'], CONFIG['height'], CONFIG['width']
    specs = (_stage_shapes(cf + cl, CONFIG['per_sub_channels'][0], CONFIG['mid_channels'][0], so)
             + _stage_shapes(cf + so + cl, CONFIG['per_sub_channels'][1], CONFIG['mid_channels'][1], so))
    keys = jax.random.split(jax.random.key(0), 2 + len(specs))
    features = jax.random.uniform(keys[0], (b, cf, h, w), dtype)
    l2out = jax.random.uniform(keys[1], (b, cl, h, w), dtype)
    params = []
    for k, (kind, shape) in zip(keys[2:], specs):
        if kind == 'a':
            params.append(jnp.full(shape, CONFIG['prelu_init'], dtype))
        else:
            params.append(jax.random.normal(k, shape, dtype) * 0.02)
    return (features, l2out, *params)


def _conv(x, w, b):
    p = w.shape[0] // 2
    y = lax.conv_general_dilated(x, w, (1, 1), [(p, p), (p, p)],
                                 dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _prelu(x, a):
    return jnp.where(x >= 0, x, a * x)


def _stage(x, p):
    i = 0
    for _ in range(CONFIG['n_concat']):
        outs = []
        for _ in range(CONFIG['n_sub']):
            x = _prelu(_conv(x, p[i], p[i + 1]), p[i + 2])
            i += 3
            outs.append(x)
        x = jnp.concatenate(outs, axis=-1)
    x = _prelu(_conv(x, p[i], p[i + 1]), p[i + 2])
    return _conv(x, p[i + 3], p[i + 4])


def workload(features, l2out, *params):
    n = len(params) // 2
    f = jnp.transpose(features, (0, 2, 3, 1))
    l2 = jnp.transpose(l2out, (0, 2, 3, 1))
    x = _stage(jnp.concatenate([f, l2], axis=-1), params[:n])
    x = jnp.concatenate([f, x, l2], axis=-1)
    out = _stage(x, params[n:])
    return jnp.transpose(out, (0, 3, 1, 2))
