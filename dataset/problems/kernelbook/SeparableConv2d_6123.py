"""[bf16 variant] Depthwise separable Conv2d: depthwise conv (groups=in_channels) + bias, then 1x1 pointwise conv + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SeparableConv2d_6123',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'depthwise_padding': 1,
    'pointwise_kernel_size': 1,
    'pointwise_padding': 0,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_dw, k_db, k_pw, k_pb = jax.random.split(key, 5)
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    k = CONFIG['kernel_size']
    pk = CONFIG['pointwise_kernel_size']
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], c_in, CONFIG['height'], CONFIG['width']), dtype
    )
    depthwise_w = jax.random.normal(k_dw, (c_in, 1, k, k), dtype) * 0.02
    depthwise_b = jax.random.normal(k_db, (c_in,), dtype) * 0.02
    pointwise_w = jax.random.normal(k_pw, (c_out, c_in, pk, pk), dtype) * 0.02
    pointwise_b = jax.random.normal(k_pb, (c_out,), dtype) * 0.02
    return x, depthwise_w, depthwise_b, pointwise_w, pointwise_b


def workload(x, depthwise_w, depthwise_b, pointwise_w, pointwise_b):
    stride = (CONFIG['stride'], CONFIG['stride'])
    dw_pad = CONFIG['depthwise_padding']
    pw_pad = CONFIG['pointwise_padding']
    dims = ('NHWC', 'OIHW', 'NHWC')

    out = jnp.transpose(x, (0, 2, 3, 1))
    out = lax.conv_general_dilated(
        out,
        depthwise_w,
        window_strides=stride,
        padding=((dw_pad, dw_pad), (dw_pad, dw_pad)),
        dimension_numbers=dims,
        feature_group_count=x.shape[1],
    )
    out = out + depthwise_b[None, None, None, :]
    out = lax.conv_general_dilated(
        out,
        pointwise_w,
        window_strides=stride,
        padding=((pw_pad, pw_pad), (pw_pad, pw_pad)),
        dimension_numbers=dims,
    )
    out = out + pointwise_b[None, None, None, :]
    return jnp.transpose(out, (0, 3, 1, 2))
