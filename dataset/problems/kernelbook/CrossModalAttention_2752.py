"""[bf16 variant] CrossModalAttention: nn.MultiheadAttention(query, key, broadcast latent value), seq-first layout."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CrossModalAttention_2752',
    'key_len': 1024,
    'query_len': 1024,
    'num_latents': 16,  # also the MHA batch dim (seq-first layout)
    'emb_dim': 1024,
    'num_heads': 4,  # must match the torch module's num_heads
}


def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    S, L = CONFIG['key_len'], CONFIG['query_len']
    N, E = CONFIG['num_latents'], CONFIG['emb_dim']
    key = jax.random.uniform(k0, (S, N, E), dtype)
    query = jax.random.uniform(k1, (L, N, E), dtype)
    latent = jax.random.normal(k2, (N, E), dtype) * 0.02
    w_in = jax.random.normal(k3, (E, 3 * E), dtype) * 0.02
    b_in = jnp.zeros((3 * E,), dtype)
    w_out = jax.random.normal(k4, (E, E), dtype) * 0.02
    b_out = jnp.zeros((E,), dtype)
    return key, query, latent, w_in, b_in, w_out, b_out


def workload(key, query, latent, w_in, b_in, w_out, b_out):
    S, N, E = key.shape
    L = query.shape[0]
    H = CONFIG['num_heads']
    D = E // H
    # value.unsqueeze(0).repeat(key.shape[0], 1, 1) -> (S, N, E)
    value = jnp.broadcast_to(latent[None], (S, N, E))
    wq, wk, wv = jnp.split(w_in, 3, axis=1)
    bq, bk, bv = jnp.split(b_in, 3)
    q = (query @ wq + bq).reshape(L, N, H, D)
    k = (key @ wk + bk).reshape(S, N, H, D)
    v = (value @ wv + bv).reshape(S, N, H, D)
    scores = jnp.einsum('lnhd,snhd->nhls', q * (D ** -0.5), k)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('nhls,snhd->lnhd', probs, v).reshape(L, N, E)
    return out @ w_out + b_out
