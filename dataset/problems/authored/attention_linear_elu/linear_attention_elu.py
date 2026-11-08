"""Non-causal linear attention: phi=elu+1 feature map, KV = phi(K)^T V, out = phi(Q) KV / (phi(Q) . sum phi(K) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'linear_attention_elu',
    'batch': 8,
    'num_heads': 16,
    'seq_len': 4096,
    'head_dim': 128,
    'v_head_dim': 128,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    b = CONFIG['batch']
    h = CONFIG['num_heads']
    s = CONFIG['seq_len']
    d = CONFIG['head_dim']
    dv = CONFIG['v_head_dim']
    q = jax.random.normal(kq, (b, h, s, d), dtype=dtype)
    k = jax.random.normal(kk, (b, h, s, d), dtype=dtype)
    v = jax.random.normal(kv, (b, h, s, dv), dtype=dtype)
    return q, k, v


def _feature_map(x):
    return jax.nn.elu(x) + 1.0


def workload(q, k, v):
    eps = CONFIG['eps']
    out_dtype = q.dtype
    qf = _feature_map(q.astype(jnp.float32))
    kf = _feature_map(k.astype(jnp.float32))
    vf = v.astype(jnp.float32)
    # Global key-value state per head: [B, H, D, Dv]
    kv_state = jnp.einsum('bhsd,bhse->bhde', kf, vf,
                          preferred_element_type=jnp.float32)
    # Key normalizer: [B, H, D]
    k_sum = jnp.sum(kf, axis=2)
    num = jnp.einsum('bhsd,bhde->bhse', qf, kv_state,
                     preferred_element_type=jnp.float32)
    den = jnp.einsum('bhsd,bhd->bhs', qf, k_sum,
                     preferred_element_type=jnp.float32)
    out = num / (den[..., None] + eps)
    return out.astype(out_dtype)
