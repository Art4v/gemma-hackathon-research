"""[bf16 variant] enhance_net_nopool (Zero-DCE++): 7x [depthwise 3x3 conv -> pointwise 1x1 conv] with ReLU/concat skips, tanh curve map, 8-step quadratic enhancement."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'enhance_net_nopool_11696',
    'batch': 8,
    'in_ch': 3,
    'height': 768,
    'width': 768,
    'number_f': 32,
    'kernel_size': 3,
    'padding': 1,
    'enhance_iters': 8,
    'scale_factor': 1.0,  # ponytail: scale 1.0 means no down/upsample; resize path omitted
}


def _layer_channels():
    c, f = CONFIG['in_ch'], CONFIG['number_f']
    return [(c, f), (f, f), (f, f), (f, f), (2 * f, f), (2 * f, f), (2 * f, c)]


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    key, kx = jax.random.split(key)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['in_ch'], CONFIG['height'], CONFIG['width']), dtype)
    k = CONFIG['kernel_size']
    params = []
    for cin, cout in _layer_channels():
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        params += [
            jax.random.normal(k1, (k, k, 1, cin), dtype) * 0.02,
            jax.random.normal(k2, (cin,), dtype) * 0.02,
            jax.random.normal(k3, (cin, cout), dtype) * 0.02,
            jax.random.normal(k4, (cout,), dtype) * 0.02,
        ]
    return (x, *params)


def _csdn(x, dw_w, dw_b, pw_w, pw_b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, dw_w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=x.shape[-1],
        precision=lax.Precision.HIGHEST) + dw_b
    return jnp.einsum('nhwc,co->nhwo', y, pw_w, precision=lax.Precision.HIGHEST) + pw_b


def workload(x, *params):
    layers = [params[i:i + 4] for i in range(0, len(params), 4)]
    h = jnp.transpose(x, (0, 2, 3, 1))
    relu = jax.nn.relu
    x1 = relu(_csdn(h, *layers[0]))
    x2 = relu(_csdn(x1, *layers[1]))
    x3 = relu(_csdn(x2, *layers[2]))
    x4 = relu(_csdn(x3, *layers[3]))
    x5 = relu(_csdn(jnp.concatenate([x3, x4], -1), *layers[4]))
    x6 = relu(_csdn(jnp.concatenate([x2, x5], -1), *layers[5]))
    x_r = jnp.tanh(_csdn(jnp.concatenate([x1, x6], -1), *layers[6]))
    x_r = jnp.transpose(x_r, (0, 3, 1, 2))
    out = x
    for _ in range(CONFIG['enhance_iters']):
        out = out + x_r * (out * out - out)
    return out, x_r
