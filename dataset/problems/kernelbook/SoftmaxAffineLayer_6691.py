"""SoftmaxAffineLayer: kernel-size-1 conv1d (TDNN affine, per-frame linear + bias) followed by log_softmax over channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftmaxAffineLayer_6691',
    'batch': 16,
    'input_dim': 1024,
    'output_dim': 1024,
    'frames': 2048,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['input_dim'], CONFIG['frames']), dtype)
    w = jax.random.normal(kw, (CONFIG['input_dim'], CONFIG['output_dim']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['output_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # Kernel size 1, no padding, stride 1: conv1d reduces to a channel matmul per frame.
    y = jnp.einsum('bit,io->bot', x, w) + b[None, :, None]
    return jax.nn.log_softmax(y, axis=CONFIG['softmax_axis'])
