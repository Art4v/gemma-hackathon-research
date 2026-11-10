"""[bf16 variant] ReflectionPad2d -> Conv2d (VALID, bias) -> InstanceNorm2d (affine) over NCHW inputs."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvLayer_17320',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'padding': 2,
    'eps': 1e-5,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jax.random.normal(
        k_w,
        (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * CONFIG['weight_scale']
    bias = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * CONFIG['weight_scale']
    scale = jnp.ones((CONFIG['out_channels'],), dtype)
    shift = jnp.zeros((CONFIG['out_channels'],), dtype)
    return x, weight, bias, scale, shift


def workload(x, weight, bias, scale, shift):
    pad = CONFIG['padding']
    stride = CONFIG['stride']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = jnp.pad(y, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')
    kernel = jnp.transpose(weight, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        y,
        kernel,
        window_strides=(stride, stride),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + bias
    mean = jnp.mean(y, axis=(1, 2), keepdims=True)
    var = jnp.var(y, axis=(1, 2), keepdims=True)
    y = (y - mean) / jnp.sqrt(var + CONFIG['eps'])
    y = y * scale + shift
    return jnp.transpose(y, (0, 3, 1, 2))
