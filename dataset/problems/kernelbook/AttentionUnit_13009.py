"""[bf16 variant] AttentionUnit: mean-variance norm -> 1x1 convs + ReLU6 -> softmax(f^T g) attention over h -> 1x1 out conv + ReLU6."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AttentionUnit_13009',
    'batch': 8,
    'channels': 256,
    'height': 64,
    'width': 64,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    B, C, H, W = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    C2 = C // 2
    keys = jax.random.split(jax.random.key(0), 10)
    fc = jax.random.uniform(keys[0], (B, C, H, W), dtype)
    fs = jax.random.uniform(keys[1], (B, C, H, W), dtype)
    f_w = jax.random.normal(keys[2], (1, 1, C, C2), dtype) * 0.02
    f_b = jax.random.normal(keys[3], (C2,), dtype) * 0.02
    g_w = jax.random.normal(keys[4], (1, 1, C, C2), dtype) * 0.02
    g_b = jax.random.normal(keys[5], (C2,), dtype) * 0.02
    h_w = jax.random.normal(keys[6], (1, 1, C, C2), dtype) * 0.02
    h_b = jax.random.normal(keys[7], (C2,), dtype) * 0.02
    out_w = jax.random.normal(keys[8], (1, 1, C2, C), dtype) * 0.02
    out_b = jax.random.normal(keys[9], (C,), dtype) * 0.02
    return fc, fs, f_w, f_b, g_w, g_b, h_w, h_b, out_w, out_b


def _mean_variance_norm(x):
    # NCHW; torch .var is unbiased (ddof=1), eps added to variance before sqrt.
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.var(x, axis=(2, 3), keepdims=True, ddof=1) + CONFIG['eps']
    return (x - mean) / jnp.sqrt(var)


def _conv1x1_relu6(x_nhwc, w_hwio, b):
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jnp.clip(y + b, 0.0, 6.0)


def workload(fc, fs, f_w, f_b, g_w, g_b, h_w, h_b, out_w, out_b):
    B, C, H, W = fc.shape
    to_nhwc = lambda t: jnp.transpose(t, (0, 2, 3, 1))
    f = _conv1x1_relu6(to_nhwc(_mean_variance_norm(fc)), f_w, f_b)
    g = _conv1x1_relu6(to_nhwc(_mean_variance_norm(fs)), g_w, g_b)
    h = _conv1x1_relu6(to_nhwc(fs), h_w, h_b)
    C2 = f.shape[-1]
    f = f.reshape(B, H * W, C2)
    g = g.reshape(B, H * W, C2)
    h = h.reshape(B, H * W, C2)
    attn = jax.nn.softmax(jnp.einsum('bic,bjc->bij', f, g), axis=-1)
    fcs = jnp.einsum('bij,bjc->bic', attn, h).reshape(B, H, W, C2)
    out = _conv1x1_relu6(fcs, out_w, out_b)
    return jnp.transpose(out, (0, 3, 1, 2))
