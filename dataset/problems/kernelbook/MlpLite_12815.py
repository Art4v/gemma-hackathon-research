"""[bf16 variant] MlpLite: avgpool3x3s2 -> permute/reshape -> fc1 -> GELU -> fc2 -> nearest upsample x2 -> conv1x1 -> permute/reshape."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MlpLite_12815',
    'batch': 16,
    'H': 64,
    'W': 64,
    'in_features': 512,
    'hidden_features': 512,
    'pool_kernel': 3,
    'pool_stride': 2,
    'pool_padding': 1,
    'upsample_scale': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    B, H, W = CONFIG['batch'], CONFIG['H'], CONFIG['W']
    C, Hd = CONFIG['in_features'], CONFIG['hidden_features']
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (B, H, W, C), dtype)
    w1 = jax.random.normal(ks[1], (C, Hd), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (Hd,), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (Hd, C), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (C,), dtype) * 0.02
    wc = jax.random.normal(ks[5], (1, 1, C, C), dtype) * 0.02  # HWIO
    bc = jax.random.normal(ks[6], (C,), dtype) * 0.02
    return x, w1, b1, w2, b2, wc, bc


def workload(x, w1, b1, w2, b2, wc, bc):
    _, H, W, C = x.shape
    k, s, p = CONFIG['pool_kernel'], CONFIG['pool_stride'], CONFIG['pool_padding']
    up = CONFIG['upsample_scale']
    x = x.reshape(-1, H, W, C)
    B = x.shape[0]
    # AvgPool2d with count_include_pad=True: divide by full window size.
    pooled = lax.reduce_window(
        x, jnp.array(0, x.dtype), lax.add,
        (1, k, k, 1), (1, s, s, 1), ((0, 0), (p, p), (p, p), (0, 0)),
    ) / (k * k)
    Ho, Wo = pooled.shape[1], pooled.shape[2]
    # torch: NCHW.permute(0,3,1,2) -> (B,Wo,C,Ho), then flat reshape.
    h = jnp.transpose(pooled, (0, 2, 3, 1)).reshape(B, Ho * Wo, C)
    h = jax.nn.gelu(h @ w1 + b1, approximate=False)
    h = h @ w2 + b2
    h = h.reshape(B, Ho, Wo, C)  # NHWC view of torch's (B,C,Ho,Wo)
    h = jnp.repeat(jnp.repeat(h, up, axis=1), up, axis=2)
    h = lax.conv_general_dilated(
        h, wc, (1, 1), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    ) + bc
    # torch: NCHW.permute(0,3,1,2) -> (B,W,C,H), then flat reshape.
    return jnp.transpose(h, (0, 2, 3, 1)).reshape(B, H * W, C)
