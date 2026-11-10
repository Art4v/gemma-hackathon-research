"""DecoderLayer: LayerNorm -> MHSA(+task_embed on q,k) -> residual, LayerNorm -> MHSA(+task_embed on q) -> residual, LayerNorm -> Linear-GELU-Linear -> residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DecoderLayer_14361',
    'batch': 8,
    'seq_len': 1024,
    'dim': 1024,
    'num_heads': 16,
    'ffn_ratio': 4.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    B, L, D, H = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim'], CONFIG['num_heads']
    F = int(D * CONFIG['ffn_ratio'])
    ks = jax.random.split(jax.random.key(0), 10)
    x = jax.random.uniform(ks[0], (B, L, D), dtype)
    task_embed = jax.random.uniform(ks[1], (1, H, L, D // H), dtype)
    w = lambda k, shape: jax.random.normal(k, shape, dtype) * 0.02
    ones, zeros = jnp.ones((D,), dtype), jnp.zeros((D,), dtype)
    return (
        x, task_embed,
        ones, zeros, w(ks[2], (D, 3 * D)), w(ks[3], (D, D)), jnp.zeros((D,), dtype),
        ones, zeros, w(ks[4], (D, 3 * D)), w(ks[5], (D, D)), jnp.zeros((D,), dtype),
        ones, zeros, w(ks[6], (D, F)), jnp.zeros((F,), dtype), w(ks[7], (F, D)), jnp.zeros((D,), dtype),
    )


def _layer_norm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['eps']) * g + b


def _attention(x, task_embed, w_qkv, w_proj, b_proj, add_to_k):
    N, L, D = x.shape
    H = task_embed.shape[1]
    Dh = D // H
    qkv = (x @ w_qkv).reshape(N, L, 3, H, Dh).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q + task_embed
    if add_to_k:
        k = k + task_embed
    attn = jnp.einsum('nhld,nhmd->nhlm', q, k) * (Dh ** -0.5)
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum('nhlm,nhmd->nhld', attn, v).transpose(0, 2, 1, 3).reshape(N, L, D)
    return out @ w_proj + b_proj


def workload(x, task_embed,
             n1_g, n1_b, qkv1_w, proj1_w, proj1_b,
             n2_g, n2_b, qkv2_w, proj2_w, proj2_b,
             n3_g, n3_b, fc1_w, fc1_b, fc2_w, fc2_b):
    x = x + _attention(_layer_norm(x, n1_g, n1_b), task_embed, qkv1_w, proj1_w, proj1_b, True)
    x = x + _attention(_layer_norm(x, n2_g, n2_b), task_embed, qkv2_w, proj2_w, proj2_b, False)
    h = jax.nn.gelu(_layer_norm(x, n3_g, n3_b) @ fc1_w + fc1_b, approximate=False)
    return x + (h @ fc2_w + fc2_b)
