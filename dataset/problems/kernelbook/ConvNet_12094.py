"""[bf16 variant] ConvNet: 5x Conv2d (ReLU after first 4) -> flatten -> log_softmax."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvNet_12094',
    'batch': 4096,
    'in_channels': 1,
    'height': 64,
    'width': 64,
    # (out_channels, kernel, stride, padding) per conv layer
    'layers': [
        (8, 5, 1, 0),
        (16, 4, 2, 1),
        (32, 5, 1, 0),
        (64, 4, 2, 1),
        (10, 4, 1, 0),
    ],
}


def create_inputs(dtype=jnp.bfloat16):
    layers = CONFIG['layers']
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(layers))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    cin = CONFIG['in_channels']
    for i, (cout, k, _, _) in enumerate(layers):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, cin, cout), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02
        params.extend([w, b])
        cin = cout
    return (x, *params)


def workload(x, *params):
    layers = CONFIG['layers']
    h = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    for i, (_, _, stride, pad) in enumerate(layers):
        w, b = params[2 * i], params[2 * i + 1]
        h = lax.conv_general_dilated(
            h, w,
            window_strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            precision=lax.Precision.HIGHEST,
        ) + b
        if i < len(layers) - 1:
            h = jax.nn.relu(h)
    h = jnp.transpose(h, (0, 3, 1, 2))  # back to NCHW so flatten order matches torch
    h = h.reshape(h.shape[0], -1)
    return jax.nn.log_softmax(h, axis=1)
