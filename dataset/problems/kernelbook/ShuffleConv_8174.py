"""ShuffleConv_8174: channel concat of r^2 copies -> pixel_shuffle(r) -> Conv2d(padding='same') with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ShuffleConv_8174',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'upscale_factor': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    r = CONFIG['upscale_factor']

    # torch.cat([X] * r**2, dim=1)
    y = jnp.concatenate([x] * (r ** 2), axis=1)

    # pixel_shuffle(y, r), NCHW
    n, c, h, wd = y.shape
    oc = c // (r ** 2)
    y = jnp.reshape(y, (n, oc, r, r, h, wd))
    y = jnp.transpose(y, (0, 1, 4, 2, 5, 3))
    y = jnp.reshape(y, (n, oc, h * r, wd * r))

    # Conv2d with padding='same': low = (k - 1) // 2, high = (k - 1) - low
    padding = []
    for k in w.shape[2:]:
        total = k - 1
        low = total // 2
        padding.append((low, total - low))

    y = jnp.transpose(y, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        w,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
