"""[bf16 variant] SelfAttnPooler: Linear(C->1) scores, masked fill -inf, softmax over time, weighted sum over time."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SelfAttnPooler_12096',
    'seq_len': 512,
    'batch': 64,
    'input_dim': 1024,
    'pad_prob': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_m, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    T, B, C = CONFIG['seq_len'], CONFIG['batch'], CONFIG['input_dim']
    encoder_out = jax.random.uniform(k_x, (T, B, C), dtype)
    # Keep t=0 unpadded so no column is fully masked (that would give NaN, as in torch).
    padding_mask = jax.random.bernoulli(k_m, CONFIG['pad_prob'], (T, B)).at[0].set(False)
    w = jax.random.normal(k_w, (C, 1), dtype) * 0.02
    b = jax.random.normal(k_b, (1,), dtype) * 0.02
    return encoder_out, padding_mask, w, b


def workload(encoder_out, padding_mask, w, b):
    x = encoder_out.astype(jnp.float32)
    logits = (jnp.matmul(x, w.astype(jnp.float32), precision=lax.Precision.HIGHEST)
              + b.astype(jnp.float32))[..., 0]
    logits = jnp.where(padding_mask, -jnp.inf, logits)
    attn = jax.nn.softmax(logits, axis=0)
    return jnp.einsum('tb,tbc->bc', attn, x, precision=lax.Precision.HIGHEST)
