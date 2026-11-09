"""TransitionUp: ConvTranspose2d(k=3, s=2, p=0) -> center crop to skip size -> concat with skip on channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TransitionUp_11466',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'in_h': 64,
    'in_w': 64,
    'skip_channels': 128,
    'skip_h': 128,
    'skip_w': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, ks, kw, kb = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['in_h'], c['in_w']), dtype)
    skip = jax.random.uniform(ks, (c['batch'], c['skip_channels'], c['skip_h'], c['skip_w']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kh, kw)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, skip, w, b


def workload(x, skip, w, b):
    k = w.shape[2]
    s = CONFIG['stride']
    pad = k - 1 - CONFIG['padding']
    # Transposed conv == conv over the stride-dilated input with a spatially flipped kernel.
    rhs = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))  # (kh, kw, Cin, Cout)
    lhs = jnp.transpose(x, (0, 2, 3, 1))
    out = lax.conv_general_dilated(
        lhs, rhs,
        window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = out + b
    out = jnp.transpose(out, (0, 3, 1, 2))
    h, wd = out.shape[2], out.shape[3]
    mh, mw = skip.shape[2], skip.shape[3]
    y0 = (h - mh) // 2
    x0 = (wd - mw) // 2
    out = out[:, :, y0:y0 + mh, x0:x0 + mw]
    return jnp.concatenate([out, skip], axis=1)
