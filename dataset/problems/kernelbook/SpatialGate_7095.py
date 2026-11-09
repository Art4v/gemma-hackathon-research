"""SpatialGate: Conv2d(1x1) gate -> ReTanH (tanh clamped at 0) -> grouped broadcast multiply with data."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialGate_7095',
    'batch': 16,
    'in_channels': 128,
    'height': 128,
    'width': 128,
    'num_groups': 1,
    'kernel_size': 1,
    'padding': 0,
    'stride': 1,
    'init_gate': 0.99,
}


def create_inputs(dtype=jnp.float32):
    b, c = CONFIG['batch'], CONFIG['in_channels']
    h, w = CONFIG['height'], CONFIG['width']
    g, k = CONFIG['num_groups'], CONFIG['kernel_size']
    k_data, k_gate, k_w = jax.random.split(jax.random.key(0), 3)
    data_input = jax.random.uniform(k_data, (b, c, h, w), dtype)
    gate_input = jax.random.uniform(k_gate, (b, c, h, w), dtype)
    weight = jax.random.normal(k_w, (k, k, c, g), dtype) * 0.02  # HWIO
    init_gate = CONFIG['init_gate']
    bias_value = 0.5 * jnp.log((1.0 + init_gate) / (1.0 - init_gate))
    bias = jnp.full((g,), bias_value, dtype)
    return data_input, gate_input, weight, bias


def workload(data_input, gate_input, weight, bias):
    s, p = CONFIG['stride'], CONFIG['padding']
    x = jnp.transpose(gate_input, (0, 2, 3, 1))  # NCHW -> NHWC
    y = lax.conv_general_dilated(
        x, weight, window_strides=(s, s), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + bias
    gate = jnp.maximum(jnp.tanh(y), 0.0)
    gate = jnp.transpose(gate, (0, 3, 1, 2))  # (B, G, Ho, Wo)
    g = weight.shape[-1]
    b = data_input.shape[0]
    data = data_input.reshape(b * g, -1, *data_input.shape[2:])
    gate = gate.reshape(b * g, -1, *gate.shape[2:])
    out = data * gate
    return out.reshape(b, -1, *out.shape[2:])
