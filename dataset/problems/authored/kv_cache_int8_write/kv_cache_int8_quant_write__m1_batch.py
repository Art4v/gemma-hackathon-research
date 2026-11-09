"""[m1 M1_batch variant of authored/kv_cache_int8_write/kv_cache_int8_quant_write] Per-head per-token absmax int8 quantize-and-write of new K/V for a leading axis of 4 layers at once, sharing the per-sequence write offsets, with batch cut from 16 to 4."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'kv_cache_int8_quant_write_m1_batch',
    'num_layers': 4,
    'batch': 4,
    'num_kv_heads': 8,
    'max_seq_len': 8192,
    'head_dim': 128,
    'new_tokens': 128,
    'qmax': 127.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    L = CONFIG['num_layers']
    B = CONFIG['batch']
    H = CONFIG['num_kv_heads']
    S = CONFIG['max_seq_len']
    D = CONFIG['head_dim']
    T = CONFIG['new_tokens']
    keys = jax.random.split(jax.random.key(0), 7)
    k_new = jax.random.normal(keys[0], (L, B, T, H, D), dtype=dtype)
    v_new = jax.random.normal(keys[1], (L, B, T, H, D), dtype=dtype)
    # One write offset per sequence, shared by every layer.
    positions = jax.random.randint(keys[2], (B,), 0, S - T + 1, dtype=jnp.int32)
    k_cache = jax.random.randint(keys[3], (L, B, H, S, D), -127, 128, dtype=jnp.int32).astype(jnp.int8)
    v_cache = jax.random.randint(keys[4], (L, B, H, S, D), -127, 128, dtype=jnp.int32).astype(jnp.int8)
    k_scale = jax.random.uniform(keys[5], (L, B, H, S), dtype=jnp.float32, minval=0.005, maxval=0.05)
    v_scale = jax.random.uniform(keys[6], (L, B, H, S), dtype=jnp.float32, minval=0.005, maxval=0.05)
    return k_new, v_new, positions, k_cache, v_cache, k_scale, v_scale


def _quantize(x):
    # x: [B, T, H, D] -> q: [B, H, T, D] int8, scale: [B, H, T] f32
    qmax = CONFIG['qmax']
    eps = CONFIG['eps']
    xf = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(xf), axis=-1)
    scale = jnp.maximum(amax, eps) / qmax
    q = jnp.clip(jnp.round(xf / scale[..., None]), -qmax, qmax).astype(jnp.int8)
    return jnp.transpose(q, (0, 2, 1, 3)), jnp.transpose(scale, (0, 2, 1))


def _write(cache, scale_cache, q, s, positions):
    def one(c, sc, qb, sb, p):
        c = lax.dynamic_update_slice(c, qb, (0, p, 0))
        sc = lax.dynamic_update_slice(sc, sb.astype(sc.dtype), (0, p))
        return c, sc
    return jax.vmap(one)(cache, scale_cache, q, s, positions)


def _layer(k_new, v_new, positions, k_cache, v_cache, k_scale, v_scale):
    kq, ks = _quantize(k_new)
    vq, vs = _quantize(v_new)
    k_cache_out, k_scale_out = _write(k_cache, k_scale, kq, ks, positions)
    v_cache_out, v_scale_out = _write(v_cache, v_scale, vq, vs, positions)
    return k_cache_out, v_cache_out, k_scale_out, v_scale_out


def workload(k_new, v_new, positions, k_cache, v_cache, k_scale, v_scale):
    # Leading layer axis on every K/V tensor; positions are shared across layers.
    return jax.vmap(_layer, in_axes=(0, 0, None, 0, 0, 0, 0))(
        k_new, v_new, positions, k_cache, v_cache, k_scale, v_scale)
