"""[bf16 variant] Conv3d x6 (VALID) with leaky_relu(0.02) between layers and a final sigmoid (3D voxel discriminator)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'discriminator_15107',
    'batch': 16,
    'in_channels': 1,
    'spatial': 128,
    'd_dim': 4,
    'z_dim': 4,
    'kernel_sizes': (4, 3, 3, 3, 3, 1),
    'strides': (1, 2, 1, 1, 1, 1),
    'negative_slope': 0.02,
}


def _channels():
    d = CONFIG['d_dim']
    return (CONFIG['in_channels'], d, d * 2, d * 4, d * 8, d * 16, CONFIG['z_dim'])


def create_inputs(dtype=jnp.bfloat16):
    n_layers = len(CONFIG['kernel_sizes'])
    keys = jax.random.split(jax.random.key(0), n_layers + 1)
    s = CONFIG['spatial']
    voxels = jax.random.uniform(
        keys[0], (CONFIG['batch'], CONFIG['in_channels'], s, s, s), dtype)
    ch = _channels()
    weights = []
    biases = []
    for i, k in enumerate(CONFIG['kernel_sizes']):
        shape = (k, k, k, ch[i], ch[i + 1])  # DHWIO
        weights.append(jax.random.normal(keys[i + 1], shape, dtype) * 0.02)
        biases.append(jnp.zeros((ch[i + 1],), dtype))
    return (voxels, *weights, *biases)


def _conv3d(x, w, b, stride):
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(stride, stride, stride),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(voxels, *params):
    n_layers = len(CONFIG['kernel_sizes'])
    weights = params[:n_layers]
    biases = params[n_layers:]
    slope = CONFIG['negative_slope']
    out = jnp.transpose(voxels, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    for i in range(n_layers):
        out = _conv3d(out, weights[i], biases[i], CONFIG['strides'][i])
        if i < n_layers - 1:
            out = jnp.where(out >= 0, out, out * slope)
    out = jax.nn.sigmoid(out)
    return jnp.transpose(out, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
