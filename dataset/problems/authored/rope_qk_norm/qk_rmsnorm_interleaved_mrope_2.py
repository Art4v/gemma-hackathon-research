"""QK-RMSNorm (per-head) followed by interleaved multimodal RoPE (Qwen3-VL MRoPE) on Q and K."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'qk_rmsnorm_interleaved_mrope',
    'batch': 2,
    'seq_len': 8192,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'mrope_section': (24, 20, 20),
    'rope_theta': 5000000.0,
    'eps': 1e-6,
    'max_position': 8192,
}


def create_inputs(dtype=jnp.bfloat16):
    B, S = CONFIG['batch'], CONFIG['seq_len']
    Hq, Hk, D = CONFIG['num_q_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kp = jax.random.split(key, 3)
    q = jax.random.normal(kq, (B, S, Hq, D), dtype=jnp.float32).astype(dtype)
    k = jax.random.normal(kk, (B, S, Hk, D), dtype=jnp.float32).astype(dtype)
    position_ids = jax.random.randint(kp, (3, B, S), 0, CONFIG['max_position'], dtype=jnp.int32)
    q_norm_w = jnp.ones((D,), dtype=jnp.float32)
    k_norm_w = jnp.ones((D,), dtype=jnp.float32)
    return q, k, position_ids, q_norm_w, k_norm_w


def _rms_norm(x, w, eps):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + eps) * w.astype(jnp.float32)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def workload(q, k, position_ids, q_norm_w, k_norm_w):
    D = q.shape[-1]
    half = D // 2
    eps = CONFIG['eps']
    theta = CONFIG['rope_theta']
    sec = CONFIG['mrope_section']

    # Per-head QK RMSNorm in f32.
    qn = _rms_norm(q, q_norm_w, eps)
    kn = _rms_norm(k, k_norm_w, eps)

    # Interleaved MRoPE angle table.
    inv_freq = 1.0 / (theta ** (jnp.arange(0, D, 2, dtype=jnp.float32) / D))  # [half]
    pos = position_ids.astype(jnp.float32)                                     # [3, B, S]
    freqs = pos[..., None] * inv_freq                                          # [3, B, S, half]
    j = jnp.arange(half)
    sel = jnp.where((j % 3 == 1) & (j < 3 * sec[1]), 1,
                    jnp.where((j % 3 == 2) & (j < 3 * sec[2]), 2, 0))          # [half]
    f = jnp.where(sel == 1, freqs[1], jnp.where(sel == 2, freqs[2], freqs[0]))  # [B, S, half]
    emb = jnp.concatenate([f, f], axis=-1)                                     # [B, S, D]
    cos = jnp.cos(emb)[:, :, None, :]
    sin = jnp.sin(emb)[:, :, None, :]

    q_rot = qn * cos + _rotate_half(qn) * sin
    k_rot = kn * cos + _rotate_half(kn) * sin
    return q_rot.astype(q.dtype), k_rot.astype(k.dtype)
