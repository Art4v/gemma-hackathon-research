"""[bf16 variant] TransformDecoder4: 9x (reflect-pad -> conv3x3 [-> relu]) with nearest 2x upsampling after conv41, conv31, conv21."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TransformDecoder4_2694',
    'batch': 8,
    'in_channels': 1024,
    'height': 64,
    'width': 64,
    # conv41, conv34, conv33, conv32, conv31, conv22, conv21, conv12, conv11
    'channels': (1024, 256, 256, 256, 256, 128, 128, 64, 64, 3),
    'kernel_size': 3,
    'pad': 1,
    'scale_factor': 2,
    'unpool_after': (0, 4, 6),
}


def create_inputs(dtype=jnp.bfloat16):
    chans = CONFIG['channels']
    n = len(chans) - 1
    k = CONFIG['kernel_size']
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n)
    y = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i in range(n):
        w = jax.random.normal(keys[1 + 2 * i], (chans[i + 1], chans[i], k, k), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params += [w, b]
    return (y, *params)


def workload(y, *params):
    p = CONFIG['pad']
    s = CONFIG['scale_factor']
    n = len(params) // 2
    x = jnp.transpose(y, (0, 2, 3, 1))
    for i in range(n):
        w, b = params[2 * i], params[2 * i + 1]
        x = jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='reflect')
        x = lax.conv_general_dilated(
            x, w, window_strides=(1, 1), padding='VALID',
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        ) + b
        if i < n - 1:
            x = jax.nn.relu(x)
        if i in CONFIG['unpool_after']:
            x = jnp.repeat(jnp.repeat(x, s, axis=1), s, axis=2)
    return jnp.transpose(x, (0, 3, 1, 2))
