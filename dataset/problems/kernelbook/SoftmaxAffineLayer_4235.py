"""[bf16 variant] Kernel-size-1 TDNN affine (conv1d == channel matmul) + bias, divide by temperature, log_softmax over channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SoftmaxAffineLayer_4235',
    'batch': 16,
    'input_dim': 1024,
    'output_dim': 1024,
    'frames': 2048,
    't': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['input_dim'], CONFIG['frames']), dtype)
    w = jax.random.normal(k_w, (CONFIG['input_dim'], CONFIG['output_dim']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['output_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # x: (B, C_in, T), w: (C_in, C_out), b: (C_out,)
    y = jnp.einsum('bct,co->bot', x, w, precision=lax.Precision.HIGHEST)
    y = (y + b[None, :, None]) / CONFIG['t']
    return jax.nn.log_softmax(y, axis=1)
