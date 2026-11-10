"""[bf16 variant] Net_6125: 5x(conv-relu) -> maxpool -> fc-relu -> 3x(conv-relu-bilinear x2) -> two 3-layer fc-relu branches -> reshape -> concat."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_6125',
    'batch': 32,
    'in_channels': 3,
    'image_size': 256,
    'encoder_channels': (32, 64, 128, 256, 512),
    'decoder_channels': (512, 512, 512),
    'kernel_size': 3,
    'padding': 1,
    'encoder_stride': 2,
    'decoder_stride': 1,
    'pool_size': 8,
    'pooled_size': 1,
    'bottleneck_channels': 256,
    'bottleneck_size': 4,
    'scale_factor': 2,
    'upsampled_size': 32,
    'branch_layers': 3,
    'branch_hidden': 32,
    'branch_frames': 100,
    'branch_channels': (3, 1),
    'frame_size': 32,
    'concat_axis': 2,
    'init_scale': 0.02,
}


def _layer_specs():
    c = CONFIG
    enc = (c['in_channels'],) + c['encoder_channels']
    dec = (c['bottleneck_channels'],) + c['decoder_channels']
    encoded = c['encoder_channels'][-1] * c['pooled_size'] ** 2
    bottleneck = c['bottleneck_channels'] * c['bottleneck_size'] ** 2
    flat = c['decoder_channels'][-1] * c['upsampled_size'] ** 2
    hidden = c['branch_hidden']
    specs = [('conv', i, o) for i, o in zip(enc[:-1], enc[1:])]
    specs.append(('linear', encoded, bottleneck))
    specs += [('conv', i, o) for i, o in zip(dec[:-1], dec[1:])]
    for ch in c['branch_channels']:
        out = c['branch_frames'] * ch * c['frame_size'] ** 2
        specs.append(('linear', flat, hidden))
        specs += [('linear', hidden, hidden)] * (c['branch_layers'] - 2)
        specs.append(('linear', hidden, out))
    return specs


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    specs = _layer_specs()
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(specs))
    x = jax.random.uniform(
        keys[0],
        (c['batch'], c['in_channels'], c['image_size'], c['image_size']),
        dtype,
    )
    arrays = [x]
    ks = c['kernel_size']
    for idx, (kind, in_dim, out_dim) in enumerate(specs):
        shape = (ks, ks, in_dim, out_dim) if kind == 'conv' else (in_dim, out_dim)
        w = jax.random.normal(keys[1 + 2 * idx], shape, dtype) * c['init_scale']
        b = jax.random.normal(keys[2 + 2 * idx], (out_dim,), dtype) * c['init_scale']
        arrays += [w, b]
    return tuple(arrays)


def _conv_relu(h, w, b, stride):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        h,
        w,
        window_strides=(stride, stride),
        padding=[(p, p), (p, p)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jax.nn.relu(y + b)


def _linear_relu(h, w, b):
    return jax.nn.relu(h @ w + b)


def _max_pool(h, size):
    window = (1, size, size, 1)
    return lax.reduce_window(h, -jnp.inf, lax.max, window, window, 'VALID')


def _upsample_axis(h, axis, scale):
    n = h.shape[axis]
    coords = (jnp.arange(n * scale, dtype=h.dtype) + 0.5) / scale - 0.5
    lower = jnp.floor(coords)
    t = coords - lower
    lower = lower.astype(jnp.int32)
    i0 = jnp.clip(lower, 0, n - 1)
    i1 = jnp.clip(lower + 1, 0, n - 1)
    shape = [1] * h.ndim
    shape[axis] = -1
    t = t.reshape(shape)
    return jnp.take(h, i0, axis=axis) * (1 - t) + jnp.take(h, i1, axis=axis) * t


def _upsample_bilinear(h, scale):
    return _upsample_axis(_upsample_axis(h, 1, scale), 2, scale)


def _nhwc_to_nchw(h):
    return jnp.transpose(h, (0, 3, 1, 2))


def _nchw_to_nhwc(h):
    return jnp.transpose(h, (0, 2, 3, 1))


def workload(*inputs):
    c = CONFIG
    x, *rest = inputs
    params = iter(rest)

    h = _nchw_to_nhwc(x)
    for _ in c['encoder_channels']:
        w, b = next(params), next(params)
        h = _conv_relu(h, w, b, c['encoder_stride'])
    h = _max_pool(h, c['pool_size'])

    encoded = c['encoder_channels'][-1] * c['pooled_size'] ** 2
    h = _nhwc_to_nchw(h).reshape(-1, encoded)
    w, b = next(params), next(params)
    h = _linear_relu(h, w, b)

    h = h.reshape(-1, c['bottleneck_channels'], c['bottleneck_size'], c['bottleneck_size'])
    h = _nchw_to_nhwc(h)
    for _ in c['decoder_channels']:
        w, b = next(params), next(params)
        h = _conv_relu(h, w, b, c['decoder_stride'])
        h = _upsample_bilinear(h, c['scale_factor'])

    flat_size = c['decoder_channels'][-1] * c['upsampled_size'] ** 2
    flat = _nhwc_to_nchw(h).reshape(-1, flat_size)

    branches = []
    for ch in c['branch_channels']:
        y = flat
        for _ in range(c['branch_layers']):
            w, b = next(params), next(params)
            y = _linear_relu(y, w, b)
        branches.append(y.reshape(-1, c['branch_frames'], ch, c['frame_size'], c['frame_size']))
    return jnp.concatenate(branches, axis=c['concat_axis'])
