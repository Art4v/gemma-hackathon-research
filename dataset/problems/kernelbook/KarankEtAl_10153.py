"""[bf16 variant] KarankEtAl: Conv3d(1x3x3)+ReLU -> Conv3d(1x3x3)+ReLU -> NCDHW flatten view -> Linear -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'KarankEtAl_10153',
    'batch': 8,
    'depth': 128,
    'height': 128,
    'width': 128,
    'input_channels': 4,
    'n_classes': 4,
    'patch_size': 5,
    'kernel': (1, 3, 3),
}


def _sizes():
    ic = CONFIG['input_channels']
    kd, kh, kw = CONFIG['kernel']
    p = CONFIG['patch_size']
    c1, c2 = 3 * ic, 9 * ic
    # Flattened size from the module's dummy pass on (1, 1, ic, p, p) through both valid convs.
    features = c2 * (ic - 2 * (kd - 1)) * (p - 2 * (kh - 1)) * (p - 2 * (kw - 1))
    return c1, c2, features, 6 * ic


def create_inputs(dtype=jnp.bfloat16):
    c1, c2, features, hidden = _sizes()
    kd, kh, kw = CONFIG['kernel']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], 1, CONFIG['depth'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (kd, kh, kw, 1, c1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c1,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (kd, kh, kw, c1, c2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c2,), dtype) * 0.02
    fc1_w = jax.random.normal(keys[5], (features, hidden), dtype) * 0.02
    fc1_b = jax.random.normal(keys[6], (hidden,), dtype) * 0.02
    fc2_w = jax.random.normal(keys[7], (hidden, CONFIG['n_classes']), dtype) * 0.02
    fc2_b = jax.random.normal(keys[8], (CONFIG['n_classes'],), dtype) * 0.02
    return x, w1, b1, w2, b2, fc1_w, fc1_b, fc2_w, fc2_b


def _conv_relu(x, w, b):
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w2, b2, fc1_w, fc1_b, fc2_w, fc2_b):
    h = jnp.transpose(x, (0, 2, 3, 4, 1))
    h = _conv_relu(h, w1, b1)
    h = _conv_relu(h, w2, b2)
    # PyTorch's view flattens NCDHW memory, so restore that order before reshaping.
    h = jnp.transpose(h, (0, 4, 1, 2, 3))
    h = h.reshape(-1, fc1_w.shape[0])
    h = h @ fc1_w + fc1_b
    return h @ fc2_w + fc2_b
