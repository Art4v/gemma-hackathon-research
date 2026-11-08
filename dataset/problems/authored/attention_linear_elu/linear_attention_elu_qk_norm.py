"""Non-causal linear attention with per-head QK RMSNorm and elu(x)+1 feature map."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'linear_attention_elu_qk_norm',
    'batch_size': 8,
    'num_heads': 16,
    'seq_len': 4096,
    'head_dim': 64,
    'eps': 1e-6,
    'norm_eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b = CONFIG['batch_size']
    h = CONFIG['num_heads']
    n = CONFIG['seq_len']
    d = CONFIG['head_dim']
    q = jax.random.normal(k1, (b, h, n, d), dtype=dtype)
    k = jax.random.normal(k2, (b, h, n, d), dtype=dtype)
    v = jax.random.normal(k3, (b, h, n, d), dtype=dtype)
    q_weight = jnp.ones((d,), dtype=dtype)
    k_weight = jnp.ones((d,), dtype=dtype)
    return q, k, v, q_weight, k_weight


def workload(q, k, v, q_weight, k_weight):
    eps = CONFIG['eps']
    norm_eps = CONFIG['norm_eps']

    qf = q.astype(jnp.float32)
    kf = k.astype(jnp.float32)
    vf = v.astype(jnp.float32)
    qw = q_weight.astype(jnp.float32)
    kw = k_weight.astype(jnp.float32)

    q_rms = jnp.sqrt(jnp.mean(qf * qf, axis=-1, keepdims=True) + norm_eps)
    k_rms = jnp.sqrt(jnp.mean(kf * kf, axis=-1, keepdims=True) + norm_eps)

    q_norm = (qf / q_rms) * qw
    k_norm = (kf / k_rms) * kw

    phi_q = jax.nn.elu(q_norm) + 1.0
    phi_k = jax.nn.elu(k_norm) + 1.0

    kv = jnp.matmul(jnp.swapaxes(phi_k, -1, -2), vf)
    num = jnp.matmul(phi_q, kv)

    k_sum = jnp.sum(phi_k, axis=-2, keepdims=True)
    den = jnp.sum(phi_q * k_sum, axis=-1, keepdims=True) + eps

    out = num / den
    return out.astype(q.dtype)
