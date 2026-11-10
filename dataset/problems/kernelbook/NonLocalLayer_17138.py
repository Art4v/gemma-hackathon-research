"""NonLocalLayer: Conv2d -> theta/phi/g Linear -> bmm affinity softmax -> bmm -> f Linear + identity residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NonLocalLayer_17138',
    'batch': 8,
    'length': 128,
    'n_agents': 16,
    'input_dim': 1024,
    'hidden_dim': 1024,
    'output_dim': 1024,
    't_kernel_size': 1,
    't_stride': 1,
    't_padding': 0,
    't_dilation': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 11)
    d_in = CONFIG['input_dim']
    d_h = CONFIG['hidden_dim']
    d_out = CONFIG['output_dim']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['length'], CONFIG['n_agents'], d_in),
        dtype,
    )
    conv_w = jax.random.normal(keys[1], (d_h, d_in, CONFIG['t_kernel_size'], 1), dtype) * scale
    conv_b = jax.random.normal(keys[2], (d_h,), dtype) * scale
    theta_w = jax.random.normal(keys[3], (d_h, d_h), dtype) * scale
    theta_b = jax.random.normal(keys[4], (d_h,), dtype) * scale
    phi_w = jax.random.normal(keys[5], (d_h, d_h), dtype) * scale
    phi_b = jax.random.normal(keys[6], (d_h,), dtype) * scale
    g_w = jax.random.normal(keys[7], (d_h, d_h), dtype) * scale
    g_b = jax.random.normal(keys[8], (d_h,), dtype) * scale
    f_w = jax.random.normal(keys[9], (d_h, d_out), dtype) * scale
    f_b = jax.random.normal(keys[10], (d_out,), dtype) * scale
    return (x, conv_w, conv_b, theta_w, theta_b, phi_w, phi_b, g_w, g_b, f_w, f_b)


def workload(x, conv_w, conv_b, theta_w, theta_b, phi_w, phi_b, g_w, g_b, f_w, f_b):
    batch, _, n_agents, _ = x.shape
    res = x
    pad = CONFIG['t_padding']
    h = lax.conv_general_dilated(
        x,
        conv_w,
        window_strides=(CONFIG['t_stride'], 1),
        padding=((pad, pad), (0, 0)),
        rhs_dilation=(CONFIG['t_dilation'], 1),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    ) + conv_b
    length = h.shape[1]
    hidden = h.shape[-1]
    n = length * n_agents
    h = h.reshape(batch, n, hidden)
    theta = h @ theta_w + theta_b
    phi = h @ phi_w + phi_b
    g = h @ g_w + g_b
    y = jax.nn.softmax((theta @ jnp.swapaxes(phi, 1, 2)) / hidden ** 0.5, axis=2)
    z = (y @ g) / n ** 0.5
    z = z @ f_w + f_b
    return z.reshape(batch, length, n_agents, f_w.shape[1]) + res
