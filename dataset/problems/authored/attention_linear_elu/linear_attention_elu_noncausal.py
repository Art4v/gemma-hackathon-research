"""Non-causal linear attention: elu+1 feature map -> phi(K)^T V state -> phi(Q) projection normalized by phi(Q).sum(phi(K))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'linear_attention_elu_noncausal',
    'batch': 4,
    'num_heads': 16,
    'seq_len': 4096,
    'head_dim': 128,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    N = CONFIG['seq_len']
    D = CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (B, H, N, D), dtype=dtype)
    k = jax.random.normal(kk, (B, H, N, D), dtype=dtype)
    v = jax.random.normal(kv, (B, H, N, D), dtype=dtype)
    return q, k, v


def _phi(x):
    return jax.nn.elu(x) + 1.0


def workload(q, k, v):
    eps = CONFIG['eps']
    out_dtype = q.dtype
    qf = _phi(q.astype(jnp.float32))
    kf = _phi(k.astype(jnp.float32))
    vf = v.astype(jnp.float32)
    # Per-head state: sum_n phi(K_n)^T V_n -> [B, H, D, Dv]
    kv_state = jnp.einsum('bhnd,bhne->bhde', kf, vf,
                          preferred_element_type=jnp.float32)
    # Normalizer: sum_n phi(K_n) -> [B, H, D]
    k_sum = jnp.sum(kf, axis=2)
    num = jnp.einsum('bhnd,bhde->bhne', qf, kv_state,
                     preferred_element_type=jnp.float32)
    den = jnp.einsum('bhnd,bhd->bhn', qf, k_sum,
                     preferred_element_type=jnp.float32)
    out = num / (den[..., None] + eps)
    return out.astype(out_dtype)
