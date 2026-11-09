"""[bf16 variant] Connection_Combination: softmax(betas, -1) -> 3*b0*x0 + 3*b1*x1 + 3*b2*x2 -> relu."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Connection_Combination_16380',
    'batch': 8,
    'channels': 64,
    'height': 128,
    'width': 256,
    'scale': 3.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    prev_parallel = jax.random.uniform(k0, shape, dtype)
    prev_above = jax.random.uniform(k1, shape, dtype)
    prev_below = jax.random.uniform(k2, shape, dtype)
    betas = jax.random.uniform(k3, shape, dtype)
    return prev_parallel, prev_above, prev_below, betas


def workload(prev_parallel, prev_above, prev_below, betas):
    s = CONFIG['scale']
    b = jax.nn.softmax(betas, axis=-1)
    # betas[i] has shape (C, H, W) and broadcasts over the batch dim, as in PyTorch.
    mix = s * b[0] * prev_parallel + s * b[1] * prev_above + s * b[2] * prev_below
    return jax.nn.relu(mix)
