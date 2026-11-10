"""Conv1d (kernel_size 1, NCL) -> ReLU -> flatten(1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvEncoder_8588',
    'batch': 64,
    'input_dim': 512,
    'output_dim': 512,
    'length': 1024,
    'kernel_size': 1,
    'stride': 1,
    'dilation': 1,
    'padding': 'VALID',
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['input_dim'], CONFIG['length']), dtype
    )
    w = jax.random.normal(
        k_w, (CONFIG['kernel_size'], CONFIG['input_dim'], CONFIG['output_dim']), dtype
    ) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['output_dim'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    y = jnp.transpose(x, (0, 2, 1))
    y = lax.conv_general_dilated(
        y,
        w,
        window_strides=(CONFIG['stride'],),
        padding=CONFIG['padding'],
        rhs_dilation=(CONFIG['dilation'],),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )
    y = y + b
    y = jnp.transpose(y, (0, 2, 1))
    y = jax.nn.relu(y)
    return y.reshape(y.shape[0], -1)
