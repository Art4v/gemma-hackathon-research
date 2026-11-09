"""[bf16 variant] LearnedUpsampling1d: ConvTranspose1d (stride == kernel_size, no bias) + per-(channel, tap) learned bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LearnedUpsampling1d_3140',
    'batch': 16,
    'in_channels': 512,
    'out_channels': 512,
    'length': 4096,
    'kernel_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    b, cin, cout = CONFIG['batch'], CONFIG['in_channels'], CONFIG['out_channels']
    length, k = CONFIG['length'], CONFIG['kernel_size']
    x = jax.random.uniform(kx, (b, cin, length), dtype)
    w = jax.random.normal(kw, (cin, cout, k), dtype) * 0.02  # PyTorch ConvTranspose1d layout (in, out, K)
    bias = jax.random.normal(kb, (cout, k), dtype) * 0.02
    return x, w, bias


def workload(x, w, bias):
    b, _, length = x.shape
    _, cout, k = w.shape
    # stride == kernel_size, no padding: windows never overlap, so the transposed conv is one einsum.
    y = jnp.einsum('bil,iok->bolk', x, w)
    y = y + bias[None, :, None, :]
    return y.reshape(b, cout, length * k)
