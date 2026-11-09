"""[bf16 variant] SAM global attention: qkv Linear -> scaled q k^T + decomposed 2D rel-pos bias (q.Rh + q.Rw) -> softmax -> attn @ v -> proj Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'sam_decomposed_relpos_attention',
    'batch': 32,
    'height': 32,
    'width': 32,
    'dim': 768,
    'num_heads': 12,
}


def create_inputs(dtype=jnp.bfloat16):
    b, h, w, c = CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['dim']
    hd = c // CONFIG['num_heads']
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.normal(ks[0], (b, h, w, c), dtype)
    w_qkv = jax.random.normal(ks[1], (c, 3 * c), dtype) * 0.02
    b_qkv = jax.random.normal(ks[2], (3 * c,), dtype) * 0.02
    w_proj = jax.random.normal(ks[3], (c, c), dtype) * 0.02
    b_proj = jax.random.normal(ks[4], (c,), dtype) * 0.02
    rel_pos_h = jax.random.normal(ks[5], (2 * h - 1, hd), dtype) * 0.02
    rel_pos_w = jax.random.normal(ks[6], (2 * w - 1, hd), dtype) * 0.02
    return x, w_qkv, b_qkv, w_proj, b_proj, rel_pos_h, rel_pos_w


def workload(x, w_qkv, b_qkv, w_proj, b_proj, rel_pos_h, rel_pos_w):
    b, h, w, c = x.shape
    nh = CONFIG['num_heads']
    hd = c // nh
    n = h * w
    scale = hd ** -0.5

    qkv = (x.reshape(b, n, c) @ w_qkv + b_qkv).reshape(b, n, 3, nh, hd)
    q, k, v = (jnp.transpose(qkv[:, :, i], (0, 2, 1, 3)) for i in range(3))  # (B, nh, N, hd)

    attn = jnp.einsum('bnqd,bnkd->bnqk', q * scale, k)

    # ponytail: F.interpolate to the same length is the identity because q_size == k_size == input_size.
    idx_h = jnp.arange(h)[:, None] - jnp.arange(h)[None, :] + (h - 1)
    idx_w = jnp.arange(w)[:, None] - jnp.arange(w)[None, :] + (w - 1)
    rh = rel_pos_h[idx_h]  # (H, H, hd)
    rw = rel_pos_w[idx_w]  # (W, W, hd)
    r_q = q.reshape(b, nh, h, w, hd)
    rel_h = jnp.einsum('bnhwc,hkc->bnhwk', r_q, rh)
    rel_w = jnp.einsum('bnhwc,wkc->bnhwk', r_q, rw)
    rel = (rel_h[..., :, None] + rel_w[..., None, :]).reshape(b, nh, n, n)

    probs = jax.nn.softmax((attn + rel).astype(jnp.float32), axis=-1).astype(q.dtype)
    out = jnp.einsum('bnqk,bnkd->bnqd', probs, v).reshape(b, nh, h, w, hd)
    out = jnp.transpose(out, (0, 2, 3, 1, 4)).reshape(b, h, w, c)
    return out @ w_proj + b_proj
