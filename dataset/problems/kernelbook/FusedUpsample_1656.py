"""FusedUpsample: equalized-lr weight scale, 2x2 blur of kernel, conv_transpose2d stride 2 + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FusedUpsample_1656',
    'batch': 16,
    'in_channel': 128,
    'out_channel': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'padding': 0,
    'stride': 2,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channel'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channel'], c['out_channel'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channel'],), dtype) * 0.02
    return x, w, b


def workload(x, weight, bias):
    cin, _, k, _ = weight.shape
    multiplier = (2.0 / (cin * k * k)) ** 0.5
    w = jnp.pad(weight * multiplier, ((0, 0), (0, 0), (1, 1), (1, 1)))
    w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) / 4
    kk = w.shape[2]
    # Transposed conv == dilated-input conv with spatially flipped kernel, HWIO with I=Cin, O=Cout.
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    pad = kk - 1 - CONFIG['padding']
    s = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    out = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    out = out + bias
    return jnp.transpose(out, (0, 3, 1, 2))
