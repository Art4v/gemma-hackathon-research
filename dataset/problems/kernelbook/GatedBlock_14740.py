"""GatedBlock: depthwise dilated Conv2d x2 -> tanh * sigmoid gate -> cropped residual add -> depthwise 1x1 Conv2d."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GatedBlock_14740',
    'batch': 8,
    'w_dim': 128,
    'height': 256,
    'width': 256,
    'gate_kernel': (2, 1),
    'gate_dilation': (1, 1),
    'out_kernel': (1, 1),
    'out_dilation': (1, 1),
    'strides': (1, 1),
    'padding': 'VALID',
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    w_dim = CONFIG['w_dim']
    scale = CONFIG['weight_scale']
    x_shape = (CONFIG['batch'], w_dim, CONFIG['height'], CONFIG['width'])
    gate_shape = (w_dim, 1) + CONFIG['gate_kernel']
    out_shape = (w_dim, 1) + CONFIG['out_kernel']
    bias_shape = (w_dim,)

    x = jax.random.uniform(keys[0], x_shape, dtype)
    tanh_w = jax.random.normal(keys[1], gate_shape, dtype) * scale
    tanh_b = jax.random.normal(keys[2], bias_shape, dtype) * scale
    sigmoid_w = jax.random.normal(keys[3], gate_shape, dtype) * scale
    sigmoid_b = jax.random.normal(keys[4], bias_shape, dtype) * scale
    out_w = jax.random.normal(keys[5], out_shape, dtype) * scale
    out_b = jax.random.normal(keys[6], bias_shape, dtype) * scale
    return x, tanh_w, tanh_b, sigmoid_w, sigmoid_b, out_w, out_b


def _depthwise_conv(x_nhwc, w, b, dilation):
    y = lax.conv_general_dilated(
        x_nhwc,
        w,
        window_strides=CONFIG['strides'],
        padding=CONFIG['padding'],
        rhs_dilation=dilation,
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        feature_group_count=x_nhwc.shape[-1],
    )
    return y + b


def workload(x, tanh_w, tanh_b, sigmoid_w, sigmoid_b, out_w, out_b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    x_tanh = _depthwise_conv(x_nhwc, tanh_w, tanh_b, CONFIG['gate_dilation'])
    x_sigmoid = _depthwise_conv(x_nhwc, sigmoid_w, sigmoid_b, CONFIG['gate_dilation'])
    x_gate = jnp.tanh(x_tanh) * jax.nn.sigmoid(x_sigmoid)
    x_res = x_gate + x_nhwc[:, :x_gate.shape[1], :, :]
    x_out = _depthwise_conv(x_res, out_w, out_b, CONFIG['out_dilation'])
    return jnp.transpose(x_out, (0, 3, 1, 2))
