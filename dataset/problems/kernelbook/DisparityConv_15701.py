"""DisparityConv: circular width shifts -> mean_c |x - shifted| -> Conv2d 3x3 (pad 1, bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DisparityConv_15701',
    'batch': 8,
    'channels': 32,
    'height': 256,
    'width': 256,
    'max_shift': 64,
    'output_nc': 128,
    'kernel_size': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    ks = CONFIG['kernel_size']
    w = jax.random.normal(k_w, (ks, ks, CONFIG['max_shift'], CONFIG['output_nc']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['output_nc'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # max_shift == conv input channels; window s of the unfold is x rolled left by s along width.
    max_shift = w.shape[2]
    diff = jnp.stack(
        [jnp.mean(jnp.abs(x - jnp.roll(x, -s, axis=-1)), axis=1) for s in range(1, max_shift + 1)],
        axis=-1,
    )  # (b, h, w, S) NHWC
    p = CONFIG['padding']
    out = lax.conv_general_dilated(
        diff, w,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    return jnp.transpose(out, (0, 3, 1, 2))
