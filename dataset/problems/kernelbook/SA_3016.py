"""SA: channel mean+max -> concat -> conv7x7(2->1) -> sigmoid -> x * attn."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SA_3016',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 7,
    'padding': 3,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    ks = CONFIG['kernel_size']
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(k_w, (1, 2, ks, ks), dtype) * 0.02  # PyTorch OIHW
    b = jax.random.normal(k_b, (1,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pad = CONFIG['padding']
    avg = jnp.mean(x, axis=1, keepdims=True)
    mx = jnp.max(x, axis=1, keepdims=True)
    feat = jnp.concatenate([avg, mx], axis=1)            # (B, 2, H, W)
    feat = jnp.transpose(feat, (0, 2, 3, 1))             # NHWC
    kernel = jnp.transpose(w, (2, 3, 1, 0))              # OIHW -> HWIO
    y = lax.conv_general_dilated(
        feat, kernel, window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + b
    attn = jnp.transpose(jax.nn.sigmoid(y), (0, 3, 1, 2))  # (B, 1, H, W)
    return x * attn
