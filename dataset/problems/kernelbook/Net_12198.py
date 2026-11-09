"""[bf16 variant] Net_12198: 12x (Conv2d + ReLU) -> NCHW view(-1, 128) -> Linear(128, 1) -> tanh."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_12198',
    'batch': 16,
    'in_channels': 5,
    'height': 512,
    'width': 512,
    # (in_channels, out_channels, kernel, stride, padding) for a1..d3
    'convs': (
        (5, 16, 3, 1, 1),
        (16, 16, 3, 1, 1),
        (16, 32, 3, 2, 0),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 64, 3, 2, 0),
        (64, 64, 2, 1, 1),
        (64, 64, 2, 1, 1),
        (64, 128, 2, 2, 0),
        (128, 128, 1, 1, 0),
        (128, 128, 1, 1, 0),
        (128, 128, 1, 1, 0),
    ),
    'flat_features': 128,
    'out_features': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    convs = CONFIG['convs']
    keys = jax.random.split(jax.random.key(0), 2 * len(convs) + 3)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i, (cin, cout, k, _, _) in enumerate(convs):
        params.append(jax.random.normal(keys[2 * i + 1], (cout, cin, k, k), dtype) * 0.02)
        params.append(jax.random.normal(keys[2 * i + 2], (cout,), dtype) * 0.02)
    w_last = jax.random.normal(
        keys[-2], (CONFIG['flat_features'], CONFIG['out_features']), dtype) * 0.02
    b_last = jax.random.normal(keys[-1], (CONFIG['out_features'],), dtype) * 0.02
    return (x, *params, w_last, b_last)


def workload(x, *params):
    convs = CONFIG['convs']
    h = jnp.transpose(x, (0, 2, 3, 1))
    for i, (_, _, _, s, p) in enumerate(convs):
        w, b = params[2 * i], params[2 * i + 1]
        h = lax.conv_general_dilated(
            h, w, window_strides=(s, s), padding=((p, p), (p, p)),
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
            precision=lax.Precision.HIGHEST,
        )
        h = jax.nn.relu(h + b)
    w_last, b_last = params[-2], params[-1]
    # PyTorch view(-1, 128) flattens contiguous NCHW memory, so go back to NCHW first.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, CONFIG['flat_features'])
    return jnp.tanh(jnp.dot(h, w_last, precision=lax.Precision.HIGHEST) + b_last)
