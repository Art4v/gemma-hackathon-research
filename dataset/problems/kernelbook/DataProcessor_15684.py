"""[bf16 variant] AdaptiveAvgPool2d((7,7)) on unbatched (C,H,W) -> squeeze -> permute(1,2,0) -> reshape(-1, C)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DataProcessor_15684',
    'channels': 1024,
    'height': 256,
    'width': 256,
    'out_h': 7,
    'out_w': 7,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype=dtype
    )
    return (x,)


def _adaptive_avg_axis(x, out_size, axis):
    # PyTorch adaptive pooling bins: [floor(i*n/out), ceil((i+1)*n/out))
    n = x.shape[axis]
    parts = []
    for i in range(out_size):
        start = (i * n) // out_size
        end = -(-((i + 1) * n) // out_size)
        sl = jax.lax.slice_in_dim(x, start, end, axis=axis)
        parts.append(jnp.mean(sl, axis=axis, keepdims=True))
    return jnp.concatenate(parts, axis=axis)


def workload(x):
    pooled = _adaptive_avg_axis(x, CONFIG['out_h'], axis=1)
    pooled = _adaptive_avg_axis(pooled, CONFIG['out_w'], axis=2)  # (C, 7, 7)
    # squeeze is a no-op for C > 1
    out = jnp.transpose(pooled, (1, 2, 0))  # (7, 7, C)
    return out.reshape(-1, out.shape[-1])
