"""[bf16 variant] LearnedUpsampling1d: ConvTranspose1d (stride == kernel_size, no bias) + learned per-(channel, tap) bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LearnedUpsampling1d_7764',
    'batch': 16,
    'in_channels': 256,
    'out_channels': 256,
    'length': 8192,
    'kernel_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    b, cin, cout = CONFIG['batch'], CONFIG['in_channels'], CONFIG['out_channels']
    length, k = CONFIG['length'], CONFIG['kernel_size']
    x = jax.random.uniform(kx, (b, cin, length), dtype)
    # PyTorch ConvTranspose1d weight layout: (in, out, K).
    w = jax.random.normal(kw, (cin, cout, k), dtype) * 0.02
    bias = jax.random.normal(kb, (cout, k), dtype) * 0.02
    return x, w, bias


def workload(x, w, bias):
    # stride == kernel_size means taps never overlap: each input position
    # produces its own K-wide output block.
    b, _, length = x.shape
    _, cout, k = w.shape
    y = jnp.einsum('bil,iok->bolk', x, w, precision=lax.Precision.HIGHEST)
    y = y + bias[None, :, None, :]
    return y.reshape(b, cout, length * k)
