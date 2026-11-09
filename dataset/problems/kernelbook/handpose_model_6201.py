"""OpenPose handpose_model: VGG trunk (conv3x3+ReLU, maxpool) -> CPM 1x1 head -> 5 refinement stages (concat, 7x7 convs, 1x1 convs)."""
import jax
import jax.numpy as jnp
from jax import lax

_NUM_PARTS = 22
_FEAT = 128

CONFIG = {
    'name': 'handpose_model_6201',
    'batch': 8,
    'in_channels': 3,
    'height': 256,
    'width': 256,
    'num_parts': _NUM_PARTS,
    'feat_channels': _FEAT,
    # conv: (name, cin, cout, k, stride, pad); pool: (name, k, stride, pad)
    'block1_0': [
        ('conv1_1', 3, 64, 3, 1, 1), ('conv1_2', 64, 64, 3, 1, 1),
        ('pool1_stage1', 2, 2, 0),
        ('conv2_1', 64, 128, 3, 1, 1), ('conv2_2', 128, 128, 3, 1, 1),
        ('pool2_stage1', 2, 2, 0),
        ('conv3_1', 128, 256, 3, 1, 1), ('conv3_2', 256, 256, 3, 1, 1),
        ('conv3_3', 256, 256, 3, 1, 1), ('conv3_4', 256, 256, 3, 1, 1),
        ('pool3_stage1', 2, 2, 0),
        ('conv4_1', 256, 512, 3, 1, 1), ('conv4_2', 512, 512, 3, 1, 1),
        ('conv4_3', 512, 512, 3, 1, 1), ('conv4_4', 512, 512, 3, 1, 1),
        ('conv5_1', 512, 512, 3, 1, 1), ('conv5_2', 512, 512, 3, 1, 1),
        ('conv5_3_CPM', 512, _FEAT, 3, 1, 1),
    ],
    'block1_1': [
        ('conv6_1_CPM', _FEAT, 512, 1, 1, 0),
        ('conv6_2_CPM', 512, _NUM_PARTS, 1, 1, 0),
    ],
    'stages': [
        [
            ('Mconv1_stage%d' % i, _NUM_PARTS + _FEAT, 128, 7, 1, 3),
            ('Mconv2_stage%d' % i, 128, 128, 7, 1, 3),
            ('Mconv3_stage%d' % i, 128, 128, 7, 1, 3),
            ('Mconv4_stage%d' % i, 128, 128, 7, 1, 3),
            ('Mconv5_stage%d' % i, 128, 128, 7, 1, 3),
            ('Mconv6_stage%d' % i, 128, 128, 1, 1, 0),
            ('Mconv7_stage%d' % i, 128, _NUM_PARTS, 1, 1, 0),
        ]
        for i in range(2, 7)
    ],
    'no_relu': ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',
                'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6'],
}


def _all_blocks():
    return [CONFIG['block1_0'], CONFIG['block1_1']] + CONFIG['stages']


def _conv_specs():
    return [l for blk in _all_blocks() for l in blk if 'pool' not in l[0]]


def create_inputs(dtype=jnp.float32):
    specs = _conv_specs()
    key = jax.random.key(0)
    keys = jax.random.split(key, 1 + 2 * len(specs))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype)
    params = []
    for i, (_, cin, cout, k, _, _) in enumerate(specs):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, cin, cout), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def _run_block(x, layers, params):
    """Apply one nn.Sequential block in NHWC; params is an iterator of (w, b)."""
    for layer in layers:
        name = layer[0]
        if 'pool' in name:
            _, k, s, p = layer
            x = lax.reduce_window(
                x, jnp.array(-jnp.inf, x.dtype), lax.max,
                (1, k, k, 1), (1, s, s, 1), [(0, 0), (p, p), (p, p), (0, 0)])
            continue
        _, _, _, _, s, p = layer
        w = next(params)
        b = next(params)
        x = lax.conv_general_dilated(
            x, w, (s, s), [(p, p), (p, p)],
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            precision=lax.Precision.HIGHEST) + b
        if name not in CONFIG['no_relu']:
            x = jax.nn.relu(x)
    return x


def workload(x, *params):
    it = iter(params)
    h = jnp.transpose(x, (0, 2, 3, 1))
    out1_0 = _run_block(h, CONFIG['block1_0'], it)
    out = _run_block(out1_0, CONFIG['block1_1'], it)
    for stage in CONFIG['stages']:
        out = _run_block(jnp.concatenate([out, out1_0], axis=-1), stage, it)
    return jnp.transpose(out, (0, 3, 1, 2))
