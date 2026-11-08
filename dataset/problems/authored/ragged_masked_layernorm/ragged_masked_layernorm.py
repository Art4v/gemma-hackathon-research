"""Ragged masked LayerNorm over variable-length padded sequences."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ragged_masked_layernorm',
    'batch_size': 32,
    'max_seq_len': 2048,
    'hidden_dim': 4096,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    S = CONFIG['max_seq_len']
    D = CONFIG['hidden_dim']
    x = jax.random.normal(k1, (B, S, D), dtype=dtype)
    seq_lens = jax.random.randint(k2, (B,), minval=1, maxval=S + 1, dtype=jnp.int32)
    weight = jnp.ones((D,), dtype=dtype)
    bias = jnp.zeros((D,), dtype=dtype)
    return x, seq_lens, weight, bias

def workload(x, seq_lens, weight, bias):
    B, S, D = x.shape
    eps = CONFIG['eps']
    
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
    normed = (x_f32 - mean) * jax.lax.rsqrt(var + eps)
    
    out = normed * weight.astype(jnp.float32) + bias.astype(jnp.float32)
    
    positions = jnp.arange(S, dtype=jnp.int32)[None, :]
    mask = (positions < seq_lens[:, None])[:, :, None]
    out = jnp.where(mask, out, 0.0)
    
    return out.astype(x.dtype)
