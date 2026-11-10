"""[bf16 variant] OpenPose bodypose: VGG conv/ReLU/max-pool backbone followed by six two-branch conv stages with channel concatenation."""
import jax
import jax.numpy as jnp
from jax import lax

# Conv layer entries are (in_channels, out_channels, kernel, stride, padding, relu); 'pool' marks a max pool.
CONFIG = {
    'name': 'bodypose_model_12060',
    'batch': 8,
    'in_channels': 3,
    'height': 512,
    'width': 512,
    'num_stages': 6,
    'pool_kernel': 2,
    'pool_stride': 2,
    'pool_padding': 0,
    'concat_axis': -1,
    # Upstream's no_relu_layers omits Mconv7_stage6_L2, so the last L2 output keeps its ReLU.
    'final_L2_relu': True,
    'backbone': [
        (3, 64, 3, 1, 1, True),
        (64, 64, 3, 1, 1, True),
        'pool',
        (64, 128, 3, 1, 1, True),
        (128, 128, 3, 1, 1, True),
        'pool',
        (128, 256, 3, 1, 1, True),
        (256, 256, 3, 1, 1, True),
        (256, 256, 3, 1, 1, True),
        (256, 256, 3, 1, 1, True),
        'pool',
        (256, 512, 3, 1, 1, True),
        (512, 512, 3, 1, 1, True),
        (512, 256, 3, 1, 1, True),
        (256, 128, 3, 1, 1, True),
    ],
    'stage1_L1': [
        (128, 128, 3, 1, 1, True),
        (128, 128, 3, 1, 1, True),
        (128, 128, 3, 1, 1, True),
        (128, 512, 1, 1, 0, True),
        (512, 38, 1, 1, 0, False),
    ],
    'stage1_L2': [
        (128, 128, 3, 1, 1, True),
        (128, 128, 3, 1, 1, True),
        (128, 128, 3, 1, 1, True),
        (128, 512, 1, 1, 0, True),
        (512, 19, 1, 1, 0, False),
    ],
    'refine_L1': [
        (185, 128, 7, 1, 3, True),
        (128, 128, 7, 1, 3, True),
        (128, 128, 7, 1, 3, True),
        (128, 128, 7, 1, 3, True),
        (128, 128, 7, 1, 3, True),
        (128, 128, 1, 1, 0, True),
        (128, 38, 1, 1, 0, False),
    ],
    'refine_L2': [
        (185, 128, 7, 1, 3, True),
        (128, 128, 7, 1, 3, True),
        (128, 128, 7, 1, 3, True),
        (128, 128, 7, 1, 3, True),
        (128, 128, 7, 1, 3, True),
        (128, 128, 1, 1, 0, True),
        (128, 19, 1, 1, 0, False),
    ],
}


def _all_blocks():
    blocks = [CONFIG['backbone'], CONFIG['stage1_L1'], CONFIG['stage1_L2']]
    for _ in range(2, CONFIG['num_stages'] + 1):
        blocks.append(CONFIG['refine_L1'])
        blocks.append(CONFIG['refine_L2'])
    return blocks


def create_inputs(dtype=jnp.bfloat16):
    shapes = []
    for block in _all_blocks():
        for layer in block:
            if layer == 'pool':
                continue
            in_channels, out_channels, kernel, _, _, _ = layer
            shapes.append((out_channels, in_channels, kernel, kernel))
            shapes.append((out_channels,))
    keys = jax.random.split(jax.random.key(0), len(shapes) + 1)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = tuple(
        jax.random.normal(keys[i + 1], shape, dtype) * 0.02
        for i, shape in enumerate(shapes)
    )
    return (x,) + params


def _max_pool(x):
    kernel = CONFIG['pool_kernel']
    stride = CONFIG['pool_stride']
    pad = CONFIG['pool_padding']
    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        (1, kernel, kernel, 1),
        (1, stride, stride, 1),
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
    )


def _run_block(x, block, params):
    for layer in block:
        if layer == 'pool':
            x = _max_pool(x)
            continue
        _, _, _, stride, pad, relu = layer
        weight = next(params)
        bias = next(params)
        x = lax.conv_general_dilated(
            x,
            weight,
            window_strides=(stride, stride),
            padding=((pad, pad), (pad, pad)),
            dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        )
        x = x + bias
        if relu:
            x = jnp.maximum(x, 0)
    return x


def workload(*inputs):
    x = jnp.transpose(inputs[0], (0, 2, 3, 1))
    params = iter(inputs[1:])
    axis = CONFIG['concat_axis']

    features = _run_block(x, CONFIG['backbone'], params)
    out_l1 = _run_block(features, CONFIG['stage1_L1'], params)
    out_l2 = _run_block(features, CONFIG['stage1_L2'], params)

    for _ in range(2, CONFIG['num_stages'] + 1):
        merged = jnp.concatenate([out_l1, out_l2, features], axis=axis)
        out_l1 = _run_block(merged, CONFIG['refine_L1'], params)
        out_l2 = _run_block(merged, CONFIG['refine_L2'], params)

    if CONFIG['final_L2_relu']:
        out_l2 = jnp.maximum(out_l2, 0)

    return jnp.transpose(out_l1, (0, 3, 1, 2)), jnp.transpose(out_l2, (0, 3, 1, 2))
