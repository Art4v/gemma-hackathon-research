"""[bf16 variant] Conv1d(k=1,3,5) branches -> ReLU -> concat -> Linear(768, 2) QA head."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ComplexCnnQAHead_16228',
    'batch': 16,
    'seq_len': 2048,
    'input_size': 1024,
    'conv_channels': 256,
    'kernel_sizes': (1, 3, 5),
    'num_outputs': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 9)
    b, l, c = CONFIG['batch'], CONFIG['seq_len'], CONFIG['input_size']
    oc, no = CONFIG['conv_channels'], CONFIG['num_outputs']
    x = jax.random.uniform(keys[0], (b, l, c), dtype)
    convs = []
    for i, k in enumerate(CONFIG['kernel_sizes']):
        w = jax.random.normal(keys[1 + 2 * i], (k, c, oc), dtype) * 0.02
        bias = jax.random.normal(keys[2 + 2 * i], (oc,), dtype) * 0.02
        convs += [w, bias]
    n_in = oc * len(CONFIG['kernel_sizes'])
    fc_w = jax.random.normal(keys[7], (n_in, no), dtype) * 0.02
    fc_b = jax.random.normal(keys[8], (no,), dtype) * 0.02
    return (x, *convs, fc_w, fc_b)


def _conv_relu(x, w, b):
    pad = w.shape[0] // 2
    y = lax.conv_general_dilated(
        x, w, window_strides=(1,), padding=[(pad, pad)],
        dimension_numbers=('NWC', 'WIO', 'NWC'))
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w3, b3, w5, b5, fc_w, fc_b):
    feats = jnp.concatenate(
        [_conv_relu(x, w1, b1), _conv_relu(x, w3, b3), _conv_relu(x, w5, b5)],
        axis=-1)
    return feats @ fc_w + fc_b
