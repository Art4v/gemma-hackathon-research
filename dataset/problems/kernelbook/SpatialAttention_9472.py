"""[bf16 variant] SpatialAttention: channel mean+max -> concat -> Conv1d(2->1, k=7) -> sigmoid -> multiply with input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialAttention_9472',
    'batch': 16,
    'channels': 128,
    'length': 32768,
    'kernel_size': 7,
    'padding': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['length']), dtype)
    # PyTorch Conv1d weight layout: (out=1, in=2, kernel)
    w = jax.random.normal(k_w, (1, 2, CONFIG['kernel_size']), dtype) * 0.02
    return x, w


def workload(x, w):
    avg_out = jnp.mean(x, axis=1, keepdims=True)
    max_out = jnp.max(x, axis=1, keepdims=True)
    pooled = jnp.concatenate([avg_out, max_out], axis=1)  # (N, 2, L)
    pooled_nlc = jnp.transpose(pooled, (0, 2, 1))           # (N, L, 2)
    w_wio = jnp.transpose(w, (2, 1, 0))                     # (K, 2, 1)
    pad = CONFIG['padding']
    conv = lax.conv_general_dilated(
        pooled_nlc, w_wio,
        window_strides=(1,),
        padding=((pad, pad),),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        precision=lax.Precision.HIGHEST,
    )                                                       # (N, L, 1)
    attn = jnp.transpose(conv, (0, 2, 1))                   # (N, 1, L)
    return jax.nn.sigmoid(attn) * x
