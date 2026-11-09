"""[bf16 variant] MiniBatchStd: global unbiased std of x, broadcast to (N,1,H,W), concat on channel dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MiniBatchStd_14366',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'ddof': 1,  # torch.std default is unbiased
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    n, _, h, w = x.shape
    std = jnp.std(x, ddof=CONFIG['ddof'])
    std_map = jnp.broadcast_to(std, (n, 1, h, w)).astype(x.dtype)
    return jnp.concatenate([x, std_map], axis=1)
