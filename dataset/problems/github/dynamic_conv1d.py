"""[bf16 variant] Dynamic depthwise causal conv1d: silu(x@w1)@w2+b -> per-token [D,W] kernels -> causal conv -> silu -> +residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'dynamic_conv1d',
    'batch': 8,
    'seq_len': 2048,
    'dim': 1024,
    'rank': 64,
    'width': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    B, T, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim']
    R, W = CONFIG['rank'], CONFIG['width']
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k1, (B, T, D), dtype)
    residual = jax.random.normal(k2, (B, T, D), dtype)
    w1 = jax.random.normal(k3, (D, R), dtype) * 0.02
    w2 = jax.random.normal(k4, (R, D * W), dtype) * 0.02
    bias = jnp.zeros((D * W,), dtype)
    return x, residual, w1, w2, bias


def workload(x, residual, w1, w2, bias):
    dtype = x.dtype
    B, T, D = x.shape
    W = w2.shape[1] // D
    z = jax.nn.silu(x @ w1)
    w = (z @ w2 + bias).reshape(B, T, D, W).astype(jnp.float32)
    xf = x.astype(jnp.float32)
    # Left-pad W-1 zeros so tap j of token t reads x[t-W+1+j] (causal).
    xpad = jnp.pad(xf, ((0, 0), (W - 1, 0), (0, 0)))
    y = sum(w[..., j] * xpad[:, j:j + T] for j in range(W))
    y = jax.nn.silu(y) + residual
    return y.astype(dtype)
