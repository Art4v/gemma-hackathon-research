"""Weight-standardized Conv2d: standardize weight per output channel, then conv2d + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv2dWS_15083',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'var_eps': 2e-05,
    'std_eps': 1e-05,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    o = w.shape[0]
    w = w - jnp.mean(w, axis=(1, 2, 3), keepdims=True)
    var = jnp.var(w.reshape(o, -1), axis=1, ddof=1)  # torch.var is unbiased by default
    std = jnp.sqrt(var + CONFIG['var_eps']).reshape(o, 1, 1, 1) + CONFIG['std_eps']
    w = w / std

    p, s, d = CONFIG['padding'], CONFIG['stride'], CONFIG['dilation']
    y = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)),
        jnp.transpose(w, (2, 3, 1, 0)),
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
