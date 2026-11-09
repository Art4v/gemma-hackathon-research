"""Pre-LN transformer block: LayerNorm -> causal MHA (fused QKV) -> residual -> LayerNorm -> MLP(GELU) -> residual, plus stacked KV."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Block_8949',
    'batch': 8,
    'seq_len': 512,
    'embedding_size': 1024,
    # Same as the torch default (4), so the port also matches the small E=4 model.
    'num_heads': 4,
    'mlp_ratio': 4,
    'eps': 1e-5,
    'mask_fill': 10000.0,
}

_P = 'highest'


def create_inputs(dtype=jnp.float32):
    b, s, e = CONFIG['batch'], CONFIG['seq_len'], CONFIG['embedding_size']
    f = e * CONFIG['mlp_ratio']
    ks = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(ks[0], (b, s, e), dtype)
    ln1_g = jnp.ones((e,), dtype)
    ln1_b = jnp.zeros((e,), dtype)
    w_qkv = jax.random.normal(ks[1], (e, 3 * e), dtype) * 0.02
    b_qkv = jax.random.normal(ks[2], (3 * e,), dtype) * 0.02
    w_o = jax.random.normal(ks[3], (e, e), dtype) * 0.02
    b_o = jax.random.normal(ks[4], (e,), dtype) * 0.02
    ln2_g = jnp.ones((e,), dtype)
    ln2_b = jnp.zeros((e,), dtype)
    w_fc1 = jax.random.normal(ks[5], (e, f), dtype) * 0.02
    b_fc1 = jax.random.normal(ks[6], (f,), dtype) * 0.02
    w_fc2 = jax.random.normal(ks[7], (f, e), dtype) * 0.02
    b_fc2 = jax.random.normal(ks[8], (e,), dtype) * 0.02
    return (x, ln1_g, ln1_b, w_qkv, b_qkv, w_o, b_o,
            ln2_g, ln2_b, w_fc1, b_fc1, w_fc2, b_fc2)


def _layernorm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['eps']) * g + b


def workload(x, ln1_g, ln1_b, w_qkv, b_qkv, w_o, b_o,
             ln2_g, ln2_b, w_fc1, b_fc1, w_fc2, b_fc2):
    bsz, s, e = x.shape
    h = CONFIG['num_heads']
    d = e // h

    qkv = jnp.matmul(_layernorm(x, ln1_g, ln1_b), w_qkv, precision=_P) + b_qkv
    q, k, v = jnp.split(qkv, 3, axis=-1)

    def heads(t):
        return jnp.transpose(t.reshape(bsz, s, h, d), (0, 2, 1, 3))

    q, k, v = heads(q), heads(k), heads(v)
    cached_kv = jnp.stack([k, v])

    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k, precision=_P) / jnp.sqrt(jnp.float32(d)).astype(x.dtype)
    mask = jnp.tril(jnp.ones((s, s), x.dtype))[None, None]
    attn = attn * mask - CONFIG['mask_fill'] * (1.0 - mask)
    attn = jax.nn.softmax(attn, axis=-1)
    y = jnp.einsum('bhqk,bhkd->bhqd', attn, v, precision=_P)
    y = jnp.transpose(y, (0, 2, 1, 3)).reshape(bsz, s, e)
    x = x + jnp.matmul(y, w_o, precision=_P) + b_o

    z = _layernorm(x, ln2_g, ln2_b)
    z = jax.nn.gelu(jnp.matmul(z, w_fc1, precision=_P) + b_fc1, approximate=False)
    z = jnp.matmul(z, w_fc2, precision=_P) + b_fc2
    return x + z, cached_kv
