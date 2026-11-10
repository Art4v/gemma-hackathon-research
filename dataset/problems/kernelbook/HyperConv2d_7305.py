"""[bf16 variant] HyperConv2d: Linear(1, P) hypernet on scalar t -> conv2d weight/bias -> conv2d(x)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'HyperConv2d_7305',
    'batch': 16,
    'dim_in': 128,
    'dim_out': 128,
    'height': 128,
    'width': 128,
    'ksize': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['ksize']
    params_dim = c['dim_out'] * c['dim_in'] * k * k + c['dim_out']
    k_t, k_x, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    t = jax.random.uniform(k_t, (1, 1), dtype)
    x = jax.random.uniform(k_x, (c['batch'], c['dim_in'], c['height'], c['width']), dtype)
    hyper_w = jax.random.normal(k_w, (1, params_dim), dtype) * 0.02
    hyper_b = jax.random.normal(k_b, (params_dim,), dtype) * 0.02
    return t, x, hyper_w, hyper_b


def workload(t, x, hyper_w, hyper_b):
    k = CONFIG['ksize']
    p = CONFIG['padding']
    s = CONFIG['stride']
    d = CONFIG['dilation']
    # ponytail: Linear with in_features=1 is an exact broadcast multiply.
    params = (t.reshape(1, 1) * hyper_w + hyper_b).reshape(-1)
    c_in = x.shape[1]
    c_out = params.shape[0] // (c_in * k * k + 1)
    weight = params[:c_out * c_in * k * k].reshape(c_out, c_in, k, k)
    # Matches PyTorch source: bias is the FIRST c_out params, not the trailing block.
    bias = params[:c_out]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + bias
    return jnp.transpose(y, (0, 3, 1, 2))
