"""[bf16 variant] ReflectionPad2d -> Conv2d (3x3, bias) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'RC_6332',
    'batch': 8,
    'in_dim': 128,
    'out_dim': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
    'activation_function': True,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (c['batch'], c['in_dim'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['kernel_size'], c['kernel_size'], c['in_dim'], c['out_dim']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['padding']
    x = jnp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='reflect')
    x = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + b
    if CONFIG['activation_function']:
        y = jax.nn.relu(y)
    return jnp.transpose(y, (0, 3, 1, 2))
