"""[bf16 variant] Forward_Grad: zero-pad right/bottom -> 2-tap forward-difference conv in x and y -> zero last column/row."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Forward_Grad_7661',
    'batch': 32,
    'channels': 1,
    'height': 1024,
    'width': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    # Fixed difference kernels from the module (not trained), same layout as PyTorch OIHW.
    wx = jnp.array([[[[-1.0, 1.0]]]], dtype=dtype)      # (1, 1, 1, 2)
    wy = jnp.array([[[[-1.0], [1.0]]]], dtype=dtype)    # (1, 1, 2, 1)
    return x, wx, wy


def workload(x, wx, wy):
    # Single-channel 2-tap cross-correlation written as an exact elementwise stencil.
    xr = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1)))
    diff_x = wx[0, 0, 0, 0] * xr[..., :-1] + wx[0, 0, 0, 1] * xr[..., 1:]
    diff_x = diff_x.at[..., -1].set(0)

    xb = jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 0)))
    diff_y = wy[0, 0, 0, 0] * xb[..., :-1, :] + wy[0, 0, 1, 0] * xb[..., 1:, :]
    diff_y = diff_y.at[..., -1, :].set(0)
    return diff_x, diff_y
