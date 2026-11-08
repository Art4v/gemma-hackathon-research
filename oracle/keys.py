"""Answer keys computed in chunks, for references whose f32 key doesn't fit on the chip (STUDY_PLAN, 2p; clanker 6710608874).

2p_GQA_Attention: the reference materialises B * Hq * S * S attention scores, 4 * 128 * 4096 * 4096 = 8.6e9, which is 34 GB in f32,
more than a v6e's 32 GB, so its f32_highest key fails and the key falls back to XLA's own bf16 output (err_xla 0 by construction).
gqa_chunked computes exactly the reference's maths one (batch, KV head) at a time: the G = Hq / Hkv query heads that share a KV
head, against that KV head, so every softmax row is computed whole (Lebron James: chunk over batch, heads or query blocks, never
over keys). Each chunk holds G * S * S scores: G = 16, so 1 GiB in f32 at full size, a few GiB with softmax temporaries. Imports
JAX only when called.
"""


def gqa_chunked(query, key, value):
    import jax, jax.numpy as jnp
    B, S, Hq, D = query.shape
    Hkv = key.shape[2]
    G = Hq // Hkv
    scale = D ** -0.5
    mask = jnp.tril(jnp.ones((S, S)))

    def one(bj):
        b, j = bj // Hkv, bj % Hkv
        q = jax.lax.dynamic_slice_in_dim(query[b], j * G, G, axis=1).transpose(1, 0, 2)  # (G, S, D)
        k, v = key[b, :, j, :], value[b, :, j, :]                                       # (S, D)
        attn = jnp.einsum("gqd,kd->gqk", q, k) * scale
        attn = jax.nn.softmax(jnp.where(mask, attn, -1e9), axis=-1)
        return jnp.einsum("gqk,kd->gqd", attn, v).transpose(1, 0, 2)                     # (S, G, D)

    out = jax.lax.map(one, jnp.arange(B * Hkv))                                          # (B * Hkv, S, G, D)
    return out.reshape(B, Hkv, S, G, D).transpose(0, 2, 1, 3, 4).reshape(B, S, Hq, D)


KEYS = {"2p_GQA_Attention": gqa_chunked}
