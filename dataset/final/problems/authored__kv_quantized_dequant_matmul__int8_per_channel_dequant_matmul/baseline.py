"""Int8 weight-only (W8A16) linear: per-output-channel dequantize int8 weights fused with bf16 activation matmul, f32 accumulation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'int8_per_channel_dequant_matmul',
    'M': 2048,        # tokens (prefill batch * seq)
    'K': 4608,        # hidden size (Gemma-2 27B)
    'N': 36864,       # intermediate size (gate / up projection)
    'qmax': 127,      # symmetric int8 range [-127, 127]
    'weight_std': 0.02,
}


def create_inputs(dtype=jnp.float32):
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    qmax = CONFIG['qmax']
    key = jax.random.key(0)
    kx, kw, ks = jax.random.split(key, 3)
    # Activations.
    x = jax.random.normal(kx, (M, K), dtype=dtype)
    # Symmetric int8 quantized weights, per-output-channel.
    w_q = jax.random.randint(kw, (N, K), -qmax, qmax + 1, dtype=jnp.int32).astype(jnp.int8)
    # Per-output-channel scales: a weight with std ~0.02 has max-abs ~ a few std,
    # so scale ~ weight_std / qmax, jittered deterministically per channel.
    jitter = jax.random.uniform(ks, (N,), dtype=jnp.float32, minval=0.5, maxval=1.5)
    w_scale = (jitter * (CONFIG['weight_std'] / qmax)).astype(jnp.float32)
    return (x, w_q, w_scale)


def workload(x, w_q, w_scale):
    # x: [M, K] activations (bf16/f32); w_q: [N, K] int8; w_scale: [N] f32.
    # Dequantize-on-load: int8 -> activation dtype is exact (|q| <= 127 fits bf16 mantissa).
    w = w_q.astype(x.dtype)
    acc = jnp.dot(x, w.T, preferred_element_type=jnp.float32)   # [M, N] f32 accumulate
    y = acc * w_scale[None, :].astype(jnp.float32)              # per-output-channel scale epilogue
    return y.astype(x.dtype)
