"""[bf16 variant] Concurrent 3D SE: spatial mean -> linear -> ReLU -> linear -> sigmoid channel gate, 1x1x1 conv3d -> sigmoid spatial gate, element-wise max of both gated tensors."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ChannelSpatialSELayer3D_13785',
    'batch': 8,
    'channels': 128,
    'depth': 8,
    'height': 128,
    'width': 128,
    'reduction_ratio': 2,
    'conv_kernel': (1, 1, 1),
    'conv_strides': (1, 1, 1),
    'conv_out_channels': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG['channels']
    r = c // CONFIG['reduction_ratio']
    o = CONFIG['conv_out_channels']
    s = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], c, CONFIG['depth'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    fc1_w = jax.random.normal(keys[1], (c, r), dtype) * s
    fc1_b = jax.random.normal(keys[2], (r,), dtype) * s
    fc2_w = jax.random.normal(keys[3], (r, c), dtype) * s
    fc2_b = jax.random.normal(keys[4], (c,), dtype) * s
    conv_w = jax.random.normal(keys[5], CONFIG['conv_kernel'] + (c, o), dtype) * s
    conv_b = jax.random.normal(keys[6], (o,), dtype) * s
    return x, fc1_w, fc1_b, fc2_w, fc2_b, conv_w, conv_b


def workload(x, fc1_w, fc1_b, fc2_w, fc2_b, conv_w, conv_b):
    # Channel SE: x is (B, C, D, H, W)
    squeeze = jnp.mean(x, axis=(2, 3, 4))
    hidden = jax.nn.relu(squeeze @ fc1_w + fc1_b)
    channel_gate = jax.nn.sigmoid(hidden @ fc2_w + fc2_b)
    channel_out = x * channel_gate[:, :, None, None, None]

    # Spatial SE: 1x1x1 conv in NDHWC, back to NCDHW for the gate
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    conv = lax.conv_general_dilated(
        x_ndhwc,
        conv_w,
        window_strides=CONFIG['conv_strides'],
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    spatial_gate = jnp.transpose(jax.nn.sigmoid(conv + conv_b), (0, 4, 1, 2, 3))
    spatial_out = x * spatial_gate

    return jnp.maximum(channel_out, spatial_out)
