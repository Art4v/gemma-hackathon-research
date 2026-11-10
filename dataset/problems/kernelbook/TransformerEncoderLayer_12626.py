"""TransformerEncoderLayer: masked multi-head self-attention, residual, LayerNorm, ReLU FFN, residual, LayerNorm, mask scaling."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TransformerEncoderLayer_12626',
    'batch': 32,
    'seq_len': 512,
    'dim': 1024,
    'n_heads': 4,
    'ffn_dim': 2048,
    'mask_keep_prob': 0.875,
    'mask_fill': -1e20,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    b, l, d, f = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim'], CONFIG['ffn_dim']
    keys = jax.random.split(jax.random.key(0), 14)

    x = jax.random.uniform(keys[0], (b, l, d), dtype)
    mask = jax.random.bernoulli(keys[1], CONFIG['mask_keep_prob'], (b, l)).astype(dtype)

    q_w = jax.random.normal(keys[2], (d, d), dtype) * 0.02
    q_b = jax.random.normal(keys[3], (d,), dtype) * 0.02
    k_w = jax.random.normal(keys[4], (d, d), dtype) * 0.02
    k_b = jax.random.normal(keys[5], (d,), dtype) * 0.02
    v_w = jax.random.normal(keys[6], (d, d), dtype) * 0.02
    v_b = jax.random.normal(keys[7], (d,), dtype) * 0.02
    out_w = jax.random.normal(keys[8], (d, d), dtype) * 0.02
    out_b = jax.random.normal(keys[9], (d,), dtype) * 0.02
    lin1_w = jax.random.normal(keys[10], (d, f), dtype) * 0.02
    lin1_b = jax.random.normal(keys[11], (f,), dtype) * 0.02
    lin2_w = jax.random.normal(keys[12], (f, d), dtype) * 0.02
    lin2_b = jax.random.normal(keys[13], (d,), dtype) * 0.02

    norm1_scale = jnp.ones((d,), dtype)
    norm1_shift = jnp.zeros((d,), dtype)
    norm2_scale = jnp.ones((d,), dtype)
    norm2_shift = jnp.zeros((d,), dtype)

    return (
        x, mask,
        q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b,
        lin1_w, lin1_b, lin2_w, lin2_b,
        norm1_scale, norm1_shift, norm2_scale, norm2_shift,
    )


def _layer_norm(x, scale, shift):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['eps']) * scale + shift


def _split_heads(t, n_heads, dim_per_head):
    b, l, _ = t.shape
    return jnp.transpose(t.reshape(b, l, n_heads, dim_per_head), (0, 2, 1, 3))


def _attention(x, mask, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b):
    b, l, d = x.shape
    n_heads = CONFIG['n_heads']
    dim_per_head = d // n_heads
    scale = dim_per_head ** 0.5

    q = _split_heads(x @ q_w + q_b, n_heads, dim_per_head)
    k = _split_heads(x @ k_w + k_b, n_heads, dim_per_head)
    v = _split_heads(x @ v_w + v_b, n_heads, dim_per_head)

    dot_prod = q @ jnp.swapaxes(k, -1, -2)
    attn_mask = mask[:, None, None, :] == 0
    dot_prod = jnp.where(attn_mask, CONFIG['mask_fill'], dot_prod)
    attn_weights = jax.nn.softmax(dot_prod / scale, axis=-1)

    attended = attn_weights @ v
    attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(b, l, d)
    return attended @ out_w + out_b


def workload(
    x, mask,
    q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b,
    lin1_w, lin1_b, lin2_w, lin2_b,
    norm1_scale, norm1_shift, norm2_scale, norm2_shift,
):
    t = x + _attention(x, mask, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b)
    t = _layer_norm(t, norm1_scale, norm1_shift)
    ffn = jax.nn.relu(t @ lin1_w + lin1_b) @ lin2_w + lin2_b
    t = t + ffn
    t = _layer_norm(t, norm2_scale, norm2_shift)
    return t * mask[..., None].astype(t.dtype)
