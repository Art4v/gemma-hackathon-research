"""[bf16 variant] YaRN RoPE (partial rotary, GQA): ramp-blended inv_freq, mscale-scaled cos/sin, NeoX rotate_half on q and k."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'yarn_rope_partial',
    'batch': 8,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'seq_len': 2048,
    'head_dim': 128,
    'rotary_dim': 64,
    'base': 10000.0,
    'factor': 4.0,
    'orig_max_pos': 4096,
    'beta_fast': 32,
    'beta_slow': 1,
    'max_pos_offset': 16384,
}


def create_inputs(dtype=jnp.bfloat16):
    kq, kk, kp = jax.random.split(jax.random.key(0), 3)
    b, t, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['head_dim']
    q = jax.random.normal(kq, (b, CONFIG['num_q_heads'], t, d), dtype)
    k = jax.random.normal(kk, (b, CONFIG['num_kv_heads'], t, d), dtype)
    offsets = jax.random.randint(kp, (b, 1), 0, CONFIG['max_pos_offset'], dtype=jnp.int32)
    position_ids = jnp.arange(t, dtype=jnp.int32)[None, :] + offsets
    return q, k, position_ids


def _yarn_inv_freq():
    dim = CONFIG['rotary_dim']
    base = CONFIG['base']
    factor = CONFIG['factor']
    omp = CONFIG['orig_max_pos']

    def correction_dim(num_rotations):
        return dim * jnp.log(omp / (num_rotations * 2 * jnp.pi)) / (2 * jnp.log(base))

    low = jnp.maximum(jnp.floor(correction_dim(CONFIG['beta_fast'])), 0.0)
    high = jnp.minimum(jnp.ceil(correction_dim(CONFIG['beta_slow'])), dim - 1.0)
    high = jnp.where(low == high, high + 0.001, high)
    ramp = jnp.clip((jnp.arange(dim // 2, dtype=jnp.float32) - low) / (high - low), 0.0, 1.0)
    pos_freqs = base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    extra = 1.0 - ramp
    return (1.0 / (factor * pos_freqs)) * (1.0 - extra) + (1.0 / pos_freqs) * extra


def _attention_factor():
    factor = CONFIG['factor']
    return 0.1 * jnp.log(factor) + 1.0 if factor > 1 else 1.0


def _apply_rope(x, cos, sin):
    r = cos.shape[-1]
    x_rot, x_pass = x[..., :r], x[..., r:]
    x1, x2 = x_rot[..., : r // 2], x_rot[..., r // 2:]
    rotated = jnp.concatenate((-x2, x1), axis=-1)
    out = x_rot * cos + rotated * sin
    return jnp.concatenate((out.astype(x.dtype), x_pass), axis=-1)


def workload(q, k, position_ids):
    inv_freq = _yarn_inv_freq()
    af = _attention_factor()
    freqs = position_ids[:, :, None].astype(jnp.float32) * inv_freq[None, None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    cos = (jnp.cos(emb) * af)[:, None, :, :]
    sin = (jnp.sin(emb) * af)[:, None, :, :]
    return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
