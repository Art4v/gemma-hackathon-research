"""[bf16 variant] Grouped multi-head attention: QKV linear -> group G tokens into features -> softmax(QK^T/sqrt(dh)) V -> output linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'grouped_token_attention',
    'batch': 8,
    'seq_len': 4096,
    'dim_model': 1024,
    'num_heads': 4,
    'group_size': 4,
    'mask_value': -1000000000.0,
}


def create_inputs(dtype=jnp.bfloat16):
    B, T, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim_model']
    ks = jax.random.split(jax.random.key(0), 11)
    q = jax.random.uniform(ks[0], (B, T, D), dtype)
    k = jax.random.uniform(ks[1], (B, T, D), dtype)
    v = jax.random.uniform(ks[2], (B, T, D), dtype)
    params = []
    for i in range(4):
        params.append(jax.random.normal(ks[3 + 2 * i], (D, D), dtype) * 0.02)
        params.append(jax.random.normal(ks[4 + 2 * i], (D,), dtype) * 0.02)
    return (q, k, v, *params)


def workload(q_in, k_in, v_in, wq, bq, wk, bk, wv, bv, wo, bo):
    H, G = CONFIG['num_heads'], CONFIG['group_size']
    B, Tq, D = q_in.shape
    Tk = k_in.shape[1]
    dh = G * D // H

    q = q_in @ wq + bq
    k = k_in @ wk + bk
    v = v_in @ wv + bv

    pad_q = (-Tq) % G
    pad_kv = (-Tk) % G
    q = jnp.pad(q, ((0, 0), (0, pad_q), (0, 0)))
    k = jnp.pad(k, ((0, 0), (0, pad_kv), (0, 0)))
    v = jnp.pad(v, ((0, 0), (0, pad_kv), (0, 0)))

    def heads(x):
        return x.reshape(B, -1, H, dh).transpose(0, 2, 1, 3)

    qh, kh, vh = heads(q), heads(k), heads(v)
    scores = jnp.einsum('bhqd,bhkd->bhqk', qh, kh) / (dh ** 0.5)
    if pad_kv:
        # Strided mask [::G] keeps token positions 0, G, 2G, ...; a group is masked if its first token is padding.
        key_pos = jnp.arange(0, Tk + pad_kv, G)
        mask = (key_pos >= Tk).astype(scores.dtype)
        scores = scores + mask[None, None, None, :] * CONFIG['mask_value']
    att_w = jax.nn.softmax(scores, axis=-1)
    o = jnp.einsum('bhqk,bhkd->bhqd', att_w, vh)
    o = o.transpose(0, 2, 1, 3).reshape(B, -1, D)
    o = o[:, :o.shape[1] - pad_q]
    o = o @ wo + bo
    return o, att_w
