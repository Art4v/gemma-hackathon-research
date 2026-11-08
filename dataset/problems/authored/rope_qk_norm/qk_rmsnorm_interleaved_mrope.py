"""QK-RMSNorm (per-head, learned weight) followed by interleaved multimodal RoPE (Qwen3-VL M-RoPE) on Q and K."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'qk_rmsnorm_interleaved_mrope',
    'batch': 2,
    'seq_len': 8192,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'mrope_section': (24, 20, 20),  # (t, h, w) frequency counts, sum = head_dim // 2
    'rope_theta': 5000000.0,
    'eps': 1e-6,
    'max_position': 8192,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    Hq = CONFIG['num_q_heads']
    Hk = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    kq, kk, kp = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (B, S, Hq, D), dtype)
    k = jax.random.normal(kk, (B, S, Hk, D), dtype)
    position_ids = jax.random.randint(
        kp, (3, B, S), 0, CONFIG['max_position'], dtype=jnp.int32)
    q_norm_weight = jnp.ones((D,), dtype)
    k_norm_weight = jnp.ones((D,), dtype)
    return q, k, position_ids, q_norm_weight, k_norm_weight


def _rms_norm(x, w, eps):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + eps) * w.astype(jnp.float32)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def workload(q, k, position_ids, q_norm_weight, k_norm_weight):
    D = q.shape[-1]
    F = D // 2
    sec_t, sec_h, sec_w = CONFIG['mrope_section']
    eps = CONFIG['eps']
    theta = CONFIG['rope_theta']

    inv_freq = 1.0 / (theta ** (jnp.arange(0, D, 2, dtype=jnp.float32) / D))  # [F]

    # Interleaved M-RoPE axis assignment per frequency index.
    idx = jnp.arange(F)
    axis_map = jnp.where((idx % 3 == 1) & (idx < 3 * sec_h), 1,
                         jnp.where((idx % 3 == 2) & (idx < 3 * sec_w), 2, 0))  # [F]

    pos = position_ids.astype(jnp.float32)            # [3, B, S]
    pos_sel = jnp.transpose(pos[axis_map], (1, 2, 0))  # [B, S, F]
    angles = pos_sel * inv_freq[None, None, :]         # [B, S, F]
    emb = jnp.concatenate([angles, angles], axis=-1)   # [B, S, D]
    cos = jnp.cos(emb)[:, :, None, :]
    sin = jnp.sin(emb)[:, :, None, :]

    qn = _rms_norm(q, q_norm_weight, eps)
    kn = _rms_norm(k, k_norm_weight, eps)

    q_rot = qn * cos + _rotate_half(qn) * sin
    k_rot = kn * cos + _rotate_half(kn) * sin
    return q_rot.astype(q.dtype), k_rot.astype(k.dtype)
