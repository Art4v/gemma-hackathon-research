"""[bf16 variant] CNN: 4x (conv2d -> relu -> maxpool2x2) -> flatten -> fc1 -> relu -> fc2 -> relu -> fc3."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CNN_17398',
    'batch': 2048,
    'in_channels': 3,
    'height': 64,
    'width': 64,
    'conv_channels': (16, 32, 64, 128),
    'conv_kernels': (5, 5, 3, 3),
    'conv_paddings': (2, 2, 1, 1),
    'pool': 2,
    'fc_dims': (128 * 4 * 4, 256, 64),
    'action_dim': 7,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 15)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    k = 1
    cin = CONFIG['in_channels']
    for cout, ks in zip(CONFIG['conv_channels'], CONFIG['conv_kernels']):
        params.append(jax.random.normal(keys[k], (ks, ks, cin, cout), dtype) * 0.02)
        params.append(jax.random.normal(keys[k + 1], (cout,), dtype) * 0.02)
        k += 2
        cin = cout
    dims = CONFIG['fc_dims'] + (CONFIG['action_dim'],)
    for din, dout in zip(dims[:-1], dims[1:]):
        params.append(jax.random.normal(keys[k], (din, dout), dtype) * 0.02)
        params.append(jax.random.normal(keys[k + 1], (dout,), dtype) * 0.02)
        k += 2
    return (x, *params)


def _conv_relu_pool(h, w, b, pad):
    p = CONFIG['pool']
    h = lax.conv_general_dilated(
        h, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    h = jax.nn.relu(h + b)
    return lax.reduce_window(
        h, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID'
    )


def workload(x, c1w, c1b, c2w, c2b, c3w, c3b, c4w, c4b,
             f1w, f1b, f2w, f2b, f3w, f3b):
    h = jnp.transpose(x, (0, 2, 3, 1))
    for (w, b), pad in zip(((c1w, c1b), (c2w, c2b), (c3w, c3b), (c4w, c4b)),
                           CONFIG['conv_paddings']):
        h = _conv_relu_pool(h, w, b, pad)
    # Back to NCHW so the flatten order matches PyTorch's view(in_size, -1).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)
    h = jax.nn.relu(h @ f1w + f1b)
    h = jax.nn.relu(h @ f2w + f2b)
    return h @ f3w + f3b
