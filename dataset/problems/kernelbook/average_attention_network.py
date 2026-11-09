"""[bf16 variant] Average attention network: cumulative-average mask matmul -> concat -> Linear(2D,2D) -> sigmoid gating."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'average_attention_network',
    'batch': 16,
    'seq_len': 2048,
    'model_dim': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    b, l, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['model_dim']
    x = jax.random.uniform(k_x, (b, l, d), dtype)
    w = jax.random.normal(k_w, (2 * d, 2 * d), dtype) * 0.02  # (in, out)
    bias = jax.random.normal(k_b, (2 * d,), dtype) * 0.02
    return x, w, bias


def workload(x, w, bias):
    seq_len = x.shape[1]
    d = x.shape[2]
    # Same mask as cumulative_average_mask: tril(ones) with row t scaled by 1/(t+1).
    triangle = jnp.tril(jnp.ones((seq_len, seq_len), dtype=x.dtype))
    inv_len = 1.0 / jnp.arange(1, seq_len + 1, dtype=x.dtype)
    mask = triangle * inv_len[:, None]
    avg = jnp.einsum('ts,bsd->btd', mask, x)
    gates = jnp.concatenate([x, avg], axis=-1) @ w + bias
    input_gate, forget_gate = gates[..., :d], gates[..., d:]
    out = jax.nn.sigmoid(input_gate) * x + jax.nn.sigmoid(forget_gate) * avg
    return out, avg
