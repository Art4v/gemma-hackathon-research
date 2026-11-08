"""Blockwise (sub-channel) quantized matmul: bf16 activations x int8 weights.

Reference for tpu_inference's blockwise quantized matmul kernel. The weight
matrix w_q [n_out, n_in] is int8 with one f32 scale per (output channel,
128-wide input block), stored as w_scale [n_in // block_size, 1, n_out]. The
activation x [n_batch, n_in] is quantized on the fly to int8 per
(row, 128-wide input block) using symmetric absmax scaling. Each block's
int8 x int8 dot (int32 result) is cast to bf16, rescaled by the activation
scale and the weight scale, and the blocks are accumulated sequentially in
bf16 before the result is cast back to the activation dtype.
"""

import jax
import jax.numpy as jnp

CONFIG = dict(
    n_batch=1024,
    n_in=4096,
    n_out=4096,
    block_size=128,
)

ACC_DTYPE = jnp.bfloat16
X_Q_DTYPE = jnp.int8
W_Q_DTYPE = jnp.int8


def create_inputs(dtype=jnp.bfloat16):
    n_batch = CONFIG["n_batch"]
    n_in = CONFIG["n_in"]
    n_out = CONFIG["n_out"]
    block_size = CONFIG["block_size"]
    n_blocks = n_in // block_size

    key = jax.random.key(0)
    kx, kw = jax.random.split(key)

    x = jax.random.normal(kx, (n_batch, n_in), dtype=jnp.float32).astype(dtype)

    # Blockwise-quantize a random weight to produce realistic w_q / w_scale.
    w = jax.random.normal(kw, (n_out, n_in), dtype=jnp.float32) * 0.02
    w_blocks = w.reshape(n_out, n_blocks, block_size)
    w_max = jnp.max(jnp.abs(w_blocks), axis=-1, keepdims=True)  # [n_out, nb, 1]
    info = jnp.iinfo(W_Q_DTYPE)
    scale = w_max / float(info.max)
    w_q = jnp.clip(jnp.round(w_blocks / scale), info.min, info.max)
    w_q = w_q.astype(W_Q_DTYPE).reshape(n_out, n_in)
    # [n_out, nb, 1] -> [nb, 1, n_out]
    w_scale = jnp.transpose(scale, (1, 2, 0)).astype(jnp.float32)
    return x, w_q, w_scale


def workload(x, w_q, w_scale):
    """x: [n_batch, n_in]; w_q: [n_out, n_in] int8; w_scale: [nb, 1, n_out] f32."""
    n_batch, n_in = x.shape
    n_out = w_q.shape[0]
    block_size = CONFIG["block_size"]
    n_blocks = n_in // block_size

    info = jnp.iinfo(X_Q_DTYPE)
    dtype_max = float(info.max)
    dtype_min = float(info.min)

    # Per (row, block) symmetric absmax quantization of the activation, in f32.
    xf = x.astype(jnp.float32).reshape(n_batch, n_blocks, block_size)
    x_max = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)  # [b, nb, 1]
    x_scale = x_max / dtype_max
    x_q = jnp.clip(jnp.round(xf / x_scale), dtype_min, dtype_max).astype(X_Q_DTYPE)

    # Blocked int8 x int8 dot with int32 result: [nb, b, n_out].
    x_q_k = jnp.transpose(x_q, (1, 0, 2))  # [nb, b, bs]
    w_q_k = jnp.transpose(w_q.reshape(n_out, n_blocks, block_size), (1, 0, 2))  # [nb, n_out, bs]
    dot = jax.lax.dot_general(
        x_q_k,
        w_q_k,
        (((2,), (2,)), ((0,), (0,))),
        preferred_element_type=jnp.int32,
    )

    lhs_scale = jnp.transpose(x_scale, (1, 0, 2)).astype(ACC_DTYPE)  # [nb, b, 1]
    rhs_scale = w_scale.astype(ACC_DTYPE)  # [nb, 1, n_out]
    res = dot.astype(ACC_DTYPE)
    res = res * lhs_scale
    res = res * rhs_scale

    acc = res[0]
    for k in range(1, n_blocks):
        acc = acc + res[k]
    return acc.astype(x.dtype)
