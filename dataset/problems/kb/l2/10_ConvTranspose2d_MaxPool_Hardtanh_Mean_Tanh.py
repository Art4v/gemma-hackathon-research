"""ConvTranspose2d, MaxPool2d, Hardtanh, spatial Mean, and Tanh."""

import jax
from jax import lax
import jax.numpy as jnp

CONFIG = {
    'name': '10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'maxpool_kernel_size': 2,
    'maxpool_stride': 2,
    'hardtanh_min': -1.0,
    'hardtanh_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
  k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
  b = CONFIG['batch_size']
  c_in = CONFIG['in_channels']
  c_out = CONFIG['out_channels']
  h = CONFIG['height']
  w = CONFIG['width']
  k = CONFIG['kernel_size']

  x = jax.random.uniform(k1, (b, c_in, h, w), dtype=dtype)
  weight = jax.random.normal(k2, (c_in, c_out, k, k), dtype=dtype) * 0.02
  bias = jax.random.normal(k3, (c_out,), dtype=dtype) * 0.02
  return x, weight, bias


def workload(x, weight, bias):
  s = CONFIG['stride']
  p = CONFIG['padding']
  k = CONFIG['kernel_size']
  mp_k = CONFIG['maxpool_kernel_size']
  mp_s = CONFIG['maxpool_stride']
  ht_min = CONFIG['hardtanh_min']
  ht_max = CONFIG['hardtanh_max']

  x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
  w_hwio = jnp.transpose(jnp.flip(weight, axis=(2, 3)), (2, 3, 0, 1))
  pad = k - 1 - p

  out = lax.conv_general_dilated(
      x_nhwc,
      w_hwio,
      window_strides=(1, 1),
      padding=((pad, pad), (pad, pad)),
      lhs_dilation=(s, s),
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
  )
  out = out + bias

  out = lax.reduce_window(
      out,
      -jnp.inf,
      lax.max,
      window_dimensions=(1, mp_k, mp_k, 1),
      window_strides=(1, mp_s, mp_s, 1),
      padding='VALID',
  )

  out = jnp.clip(out, ht_min, ht_max)
  out = jnp.transpose(out, (0, 3, 1, 2))
  out = jnp.mean(out, axis=(2, 3), keepdims=True)
  out = jnp.tanh(out)
  return out
