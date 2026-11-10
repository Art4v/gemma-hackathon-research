"""[bf16 variant] BottleneckBlock_7: conv1x1 -> relu -> conv3x3(stride 2, bias) -> relu -> conv1x1 -> add identity shortcut (broadcast) -> relu."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BottleneckBlock_7',
    'batch': 8192,
    'indim': 1024,
    'outdim': 1024,
    'bottleneck_dim': 256,
    'height': 2,
    'width': 2,
    'c1_kernel': 1,
    'c2_kernel': 3,
    'c3_kernel': 1,
    'c2_stride': 2,
    'c2_padding': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_c1, k_c2, k_b2, k_c3 = jax.random.split(key, 5)
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['indim'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    c1_w = jax.random.normal(
        k_c1,
        (CONFIG['bottleneck_dim'], CONFIG['indim'], CONFIG['c1_kernel'], CONFIG['c1_kernel']),
        dtype,
    ) * scale
    c2_w = jax.random.normal(
        k_c2,
        (CONFIG['bottleneck_dim'], CONFIG['bottleneck_dim'], CONFIG['c2_kernel'], CONFIG['c2_kernel']),
        dtype,
    ) * scale
    c2_b = jax.random.normal(k_b2, (CONFIG['bottleneck_dim'],), dtype) * scale
    c3_w = jax.random.normal(
        k_c3,
        (CONFIG['outdim'], CONFIG['bottleneck_dim'], CONFIG['c3_kernel'], CONFIG['c3_kernel']),
        dtype,
    ) * scale
    return x, c1_w, c2_w, c2_b, c3_w


def workload(x, c1_w, c2_w, c2_b, c3_w):
    dims = ('NHWC', 'OIHW', 'NHWC')
    no_pad = ((0, 0), (0, 0))
    pad = CONFIG['c2_padding']
    stride = CONFIG['c2_stride']

    # NCHW -> NHWC; the identity shortcut keeps full spatial resolution.
    short_out = jnp.transpose(x, (0, 2, 3, 1))

    out = lax.conv_general_dilated(
        short_out, c1_w, window_strides=(1, 1), padding=no_pad, dimension_numbers=dims
    )
    out = jax.nn.relu(out)
    out = lax.conv_general_dilated(
        out,
        c2_w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=dims,
    ) + c2_b
    out = jax.nn.relu(out)
    out = lax.conv_general_dilated(
        out, c3_w, window_strides=(1, 1), padding=no_pad, dimension_numbers=dims
    )
    # Strided path is (B, 1, 1, C) and broadcasts against the (B, 2, 2, C) shortcut.
    out = out + short_out
    out = jax.nn.relu(out)
    return jnp.transpose(out, (0, 3, 1, 2))
