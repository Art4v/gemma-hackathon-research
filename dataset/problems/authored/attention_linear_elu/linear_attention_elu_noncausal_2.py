"""Non-causal linear attention: phi=elu+1, out = phi(Q)(phi(K)^T V) / (phi(Q) . sum_n phi(K) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'linear_attention_elu_noncausal',
    'batch': 8,
    'num_heads': 16,
    'seq_len': 4096,
    'head_dim': 128,
    'v_head_dim': 128,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    N = CONFIG['seq_len']
    D = CONFIG['head_dim']
    Dv = CONFIG['v_head_dim']
    q = jax.random.normal(kq, (B, H, N, D), dtype=dtype)
    k = jax.random.normal(kk, (B, H, N, D), dtype=dtype)
    v = jax.random.normal(kv, (B, H, N, Dv), dtype=dtype)
    return q, k, v


def _feature_map(x):
    return jax.nn.elu(x) + 1.0


def workload(q, k, v):
    eps = CONFIG['eps']
    out_dtype = q.dtype
    phi_q = _feature_map(q.astype(jnp.float32))
    phi_k = _feature_map(k.astype(jnp.float32))
    v32 = v.astype(jnp.float32)
    # Summarize keys/values over the full sequence: [B,H,D,Dv] and [B,H,D].
    kv_state = jnp.einsum('bhnd,bhne->bhde', phi_k, v32,
                          preferred_element_type=jnp.float32)
    k_sum = jnp.sum(phi_k, axis=2)
    # Project queries onto the state and normalize.
    num = jnp.einsum('bhnd,bhde->bhne', phi_q, kv_state,
                     preferred_element_type=jnp.float32)
    den = jnp.einsum('bhnd,bhd->bhn', phi_q, k_sum,
                     preferred_element_type=jnp.float32)
    out = num / (den[..., None] + eps)
    return out.astype(out_dtype)
