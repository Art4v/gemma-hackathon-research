"""[bf16 variant] StageBlock: 5 x (3 chained Conv3x3+PReLU, channel concat) -> Conv1x1+PReLU -> Conv1x1."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'StageBlock_5368',
    'batch': 8,
    'in_channels': 128,
    'inner_channels': 128,
    'innerout_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'num_stages': 5,
    'convs_per_stage': 3,
    'kernel_size': 3,
    'prelu_init': 0.25,
}


def _layer_specs():
    """(cin, cout, k, has_prelu) for every conv in forward order."""
    c = CONFIG
    inner, n = c['inner_channels'], c['convs_per_stage']
    k = c['kernel_size']
    specs = []
    for s in range(c['num_stages']):
        first_in = c['in_channels'] if s == 0 else inner * n
        specs.append((first_in, inner, k, True))
        specs.extend((inner, inner, k, True) for _ in range(n - 1))
    specs.append((inner * n, c['innerout_channels'], 1, True))
    specs.append((c['innerout_channels'], c['out_channels'], 1, False))
    return specs


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    specs = _layer_specs()
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(specs))
    x = jax.random.uniform(
        keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    params = []
    for i, (cin, cout, k, has_prelu) in enumerate(specs):
        params.append(jax.random.normal(keys[1 + 2 * i], (cout, cin, k, k), dtype) * 0.02)
        params.append(jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02)
        if has_prelu:
            params.append(jnp.full((cout,), c['prelu_init'], dtype))
    return (x, *params)


def _conv(h, w, b):
    pad = w.shape[2] // 2
    y = lax.conv_general_dilated(
        h, w, window_strides=(1, 1), padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def _conv_prelu(h, w, b, a):
    y = _conv(h, w, b)
    return jnp.where(y >= 0, y, a * y)


def workload(x, *params):
    n = CONFIG['convs_per_stage']
    h = jnp.transpose(x, (0, 2, 3, 1))
    pos = 0
    for _ in range(CONFIG['num_stages']):
        outs = []
        for _ in range(n):
            w, b, a = params[pos:pos + 3]
            pos += 3
            h = _conv_prelu(h, w, b, a)
            outs.append(h)
        h = jnp.concatenate(outs, axis=-1)
    w, b, a = params[pos:pos + 3]
    h = _conv_prelu(h, w, b, a)
    w, b = params[pos + 3:pos + 5]
    h = _conv(h, w, b)
    return jnp.transpose(h, (0, 3, 1, 2))
