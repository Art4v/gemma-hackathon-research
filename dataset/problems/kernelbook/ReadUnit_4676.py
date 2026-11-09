"""ReadUnit: Linear(m_prev) * k -> concat Linear -> scale by c[-1] -> attn Linear -> softmax over N -> weighted sum of k."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ReadUnit_4676',
    'steps': 4,
    'batch': 16,
    'dim': 512,
    'knowledge': 4096,
}


def create_inputs(dtype=jnp.float32):
    t, b, d, n = CONFIG['steps'], CONFIG['batch'], CONFIG['dim'], CONFIG['knowledge']
    k_mem, k_know, k_ctrl, k_w1, k_w2, k_w3 = jax.random.split(jax.random.key(0), 6)
    memories = jax.random.uniform(k_mem, (t, b, d), dtype)
    know = jax.random.uniform(k_know, (b, d, n), dtype)
    control = jax.random.uniform(k_ctrl, (t, n), dtype)
    mem_w = jax.random.normal(k_w1, (d, d), dtype) * 0.02
    mem_b = jnp.zeros((d,), dtype)
    concat_w = jax.random.normal(k_w2, (2 * d, d), dtype) * 0.02
    concat_b = jnp.zeros((d,), dtype)
    attn_w = jax.random.normal(k_w3, (d, 1), dtype) * 0.02
    attn_b = jnp.zeros((1,), dtype)
    return (memories, know, control, mem_w, mem_b, concat_w, concat_b, attn_w, attn_b)


def workload(memories, know, control, mem_w, mem_b, concat_w, concat_b, attn_w, attn_b):
    m_prev = memories[-1] @ mem_w + mem_b                          # (B, D)
    inter = m_prev[:, :, None] * know                              # (B, D, N)
    cat = jnp.transpose(jnp.concatenate([inter, know], axis=1), (0, 2, 1))  # (B, N, 2D)
    inter = cat @ concat_w + concat_b                              # (B, N, D)
    # c[-1].unsqueeze(1) is (N, 1) in torch: scales each knowledge position.
    attn = inter * control[-1][:, None]
    logits = (attn @ attn_w + attn_b)[..., 0]                      # (B, N)
    probs = jax.nn.softmax(logits, axis=1)
    return jnp.sum(probs[:, None, :] * know, axis=2)               # (B, D)
