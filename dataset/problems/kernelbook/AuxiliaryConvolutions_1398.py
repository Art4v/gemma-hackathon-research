"""[bf16 variant] SSD auxiliary convolutions: eight Conv2d+ReLU layers returning the conv8_2, conv9_2, conv10_2 and conv11_2 feature maps."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AuxiliaryConvolutions_1398',
    'batch': 8,
    'channels': 1024,
    'height': 128,
    'width': 128,
    'weight_scale': 0.02,
    # (in_channels, out_channels, kernel_size, stride, padding)
    'layers': (
        (1024, 256, 1, 1, 0),  # conv8_1
        (256, 512, 3, 2, 1),   # conv8_2
        (512, 128, 1, 1, 0),   # conv9_1
        (128, 256, 3, 2, 1),   # conv9_2
        (256, 128, 1, 1, 0),   # conv10_1
        (128, 256, 3, 1, 0),   # conv10_2
        (256, 128, 1, 1, 0),   # conv11_1
        (128, 256, 3, 1, 0),   # conv11_2
    ),
    # Indices into 'layers' whose activations are returned.
    'output_layers': (1, 3, 5, 7),
}


def create_inputs(dtype=jnp.bfloat16):
    layers = CONFIG['layers']
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(layers))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i, (in_ch, out_ch, kernel, _, _) in enumerate(layers):
        w = jax.random.normal(keys[1 + 2 * i], (out_ch, in_ch, kernel, kernel), dtype) * CONFIG['weight_scale']
        b = jax.random.normal(keys[2 + 2 * i], (out_ch,), dtype) * CONFIG['weight_scale']
        params.extend([w, b])
    return (x, *params)


def _conv(x, w, b, stride, padding):
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding=[(padding, padding), (padding, padding)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def workload(x, *params):
    out = jnp.transpose(x, (0, 2, 3, 1))
    outputs = []
    for i, (_, _, _, stride, padding) in enumerate(CONFIG['layers']):
        out = jax.nn.relu(_conv(out, params[2 * i], params[2 * i + 1], stride, padding))
        if i in CONFIG['output_layers']:
            outputs.append(jnp.transpose(out, (0, 3, 1, 2)))
    return tuple(outputs)
