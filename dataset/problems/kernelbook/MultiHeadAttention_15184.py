"""MultiHeadAttention: linear q/k/v projections, head split, scaled dot-product attention with additive mask and softmax, head merge, dense projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiHeadAttention_15184',
    'batch': 8,
    'groups': 4,
    'seq_len': 256,
    'mask_size': 1024,
    'd_model': 256,
    'num_heads': 8,
    'mask_penalty': -1000000000.0,
    'init_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 12)
    batch = CONFIG['batch']
    groups = CONFIG['groups']
    seq_len = CONFIG['seq_len']
    mask_size = CONFIG['mask_size']
    d_model = CONFIG['d_model']
    num_heads = CONFIG['num_heads']
    scale = CONFIG['init_scale']
    activation_shape = (batch, groups, seq_len, d_model)
    weight_shape = (d_model, d_model)
    bias_shape = (d_model,)
    v = jax.random.uniform(keys[0], activation_shape, dtype)
    k = jax.random.uniform(keys[1], activation_shape, dtype)
    q = jax.random.uniform(keys[2], activation_shape, dtype)
    mask = jax.random.uniform(keys[3], (batch, num_heads, mask_size, mask_size), dtype)
    wq_w = jax.random.normal(keys[4], weight_shape, dtype) * scale
    wq_b = jax.random.normal(keys[5], bias_shape, dtype) * scale
    wk_w = jax.random.normal(keys[6], weight_shape, dtype) * scale
    wk_b = jax.random.normal(keys[7], bias_shape, dtype) * scale
    wv_w = jax.random.normal(keys[8], weight_shape, dtype) * scale
    wv_b = jax.random.normal(keys[9], bias_shape, dtype) * scale
    dense_w = jax.random.normal(keys[10], weight_shape, dtype) * scale
    dense_b = jax.random.normal(keys[11], bias_shape, dtype) * scale
    return (v, k, q, mask, wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, dense_w, dense_b)


def workload(*inputs):
    v, k, q, mask, wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, dense_w, dense_b = inputs
    batch = q.shape[0]
    d_model = dense_w.shape[0]
    num_heads = mask.shape[1]
    depth = d_model // num_heads

    def split_into_heads(x):
        return jnp.transpose(x.reshape(batch, -1, num_heads, depth), (0, 2, 1, 3))

    q_heads = split_into_heads(q @ wq_w + wq_b)
    k_heads = split_into_heads(k @ wk_w + wk_b)
    v_heads = split_into_heads(v @ wv_w + wv_b)

    logits = q_heads @ jnp.swapaxes(k_heads, -1, -2)
    logits = logits / jnp.sqrt(jnp.asarray(k_heads.shape[-1], dtype=q.dtype))
    logits = logits + mask * CONFIG['mask_penalty']
    attention_weights = jax.nn.softmax(logits, axis=-1)
    attended = attention_weights @ v_heads

    merged = jnp.transpose(attended, (0, 2, 1, 3)).reshape(batch, -1, d_model)
    return merged @ dense_w + dense_b
