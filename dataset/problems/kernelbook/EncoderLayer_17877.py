"""EncoderLayer: LayerNorm -> multi-head self-attention (softmax) -> residual -> LayerNorm -> Linear-GELU-Linear FFN -> residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EncoderLayer_17877',
    'batch': 8,
    'seq_len': 1024,
    'hidden_size': 2048,
    'ffn_size': 2048,
    'num_heads': 4,
    'eps': 1e-5,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    batch = CONFIG['batch']
    seq_len = CONFIG['seq_len']
    hidden = CONFIG['hidden_size']
    ffn = CONFIG['ffn_size']
    s = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 13)
    x = jax.random.uniform(keys[0], (batch, seq_len, hidden), dtype)
    norm1_scale = jnp.ones((hidden,), dtype)
    norm1_shift = jnp.zeros((hidden,), dtype)
    q_w = jax.random.normal(keys[1], (hidden, hidden), dtype) * s
    q_b = jax.random.normal(keys[2], (hidden,), dtype) * s
    k_w = jax.random.normal(keys[3], (hidden, hidden), dtype) * s
    k_b = jax.random.normal(keys[4], (hidden,), dtype) * s
    v_w = jax.random.normal(keys[5], (hidden, hidden), dtype) * s
    v_b = jax.random.normal(keys[6], (hidden,), dtype) * s
    out_w = jax.random.normal(keys[7], (hidden, hidden), dtype) * s
    out_b = jax.random.normal(keys[8], (hidden,), dtype) * s
    norm2_scale = jnp.ones((hidden,), dtype)
    norm2_shift = jnp.zeros((hidden,), dtype)
    ffn1_w = jax.random.normal(keys[9], (hidden, ffn), dtype) * s
    ffn1_b = jax.random.normal(keys[10], (ffn,), dtype) * s
    ffn2_w = jax.random.normal(keys[11], (ffn, hidden), dtype) * s
    ffn2_b = jax.random.normal(keys[12], (hidden,), dtype) * s
    return (
        x,
        norm1_scale, norm1_shift,
        q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b,
        norm2_scale, norm2_shift,
        ffn1_w, ffn1_b, ffn2_w, ffn2_b,
    )


def _layer_norm(x, scale, shift):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['eps']) * scale + shift


def _linear(x, w, b):
    return x @ w + b


def workload(x, norm1_scale, norm1_shift, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b,
             norm2_scale, norm2_shift, ffn1_w, ffn1_b, ffn2_w, ffn2_b):
    batch, seq_len, hidden = x.shape
    num_heads = CONFIG['num_heads']
    att_size = hidden // num_heads
    scale = att_size ** -0.5

    y = _layer_norm(x, norm1_scale, norm1_shift)
    q = _linear(y, q_w, q_b).reshape(batch, seq_len, num_heads, att_size)
    k = _linear(y, k_w, k_b).reshape(batch, seq_len, num_heads, att_size)
    v = _linear(y, v_w, v_b).reshape(batch, seq_len, num_heads, att_size)
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 3, 1))
    v = jnp.transpose(v, (0, 2, 1, 3))

    attn_mat = jax.nn.softmax((q * scale) @ k, axis=-1)
    y = attn_mat @ v
    y = jnp.transpose(y, (0, 2, 1, 3)).reshape(batch, seq_len, num_heads * att_size)
    y = _linear(y, out_w, out_b)
    x = x + y

    y = _layer_norm(x, norm2_scale, norm2_shift)
    y = jax.nn.gelu(_linear(y, ffn1_w, ffn1_b), approximate=False)
    y = _linear(y, ffn2_w, ffn2_b)
    x = x + y
    return x, attn_mat
