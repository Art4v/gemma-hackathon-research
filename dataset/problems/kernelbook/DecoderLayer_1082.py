"""DecoderLayer: masked self-MHA, add+LayerNorm, cross-MHA, add+LayerNorm, Linear-ReLU-Linear FFN, add+LayerNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DecoderLayer_1082',
    'batch': 8,
    'dec_seq_len': 512,
    'enc_seq_len': 512,
    'model_dim': 1024,
    'key_dim': 1024,
    'value_dim': 1024,
    'hidden_dim': 4096,
    'num_head': 16,
    'ln_eps': 1e-5,
    'mask_fill': -1000000000.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    b, sd, se = c['batch'], c['dec_seq_len'], c['enc_seq_len']
    d, kd, vd, hd, nh = c['model_dim'], c['key_dim'], c['value_dim'], c['hidden_dim'], c['num_head']
    keys = jax.random.split(jax.random.key(0), 24)
    dec = jax.random.uniform(keys[0], (b, sd, d), dtype)
    enc = jax.random.uniform(keys[1], (b, se, d), dtype)
    source_mask = jax.random.uniform(keys[2], (b, nh, sd, se), dtype)
    target_mask = jax.random.uniform(keys[3], (b, nh, sd, sd), dtype)

    def w(k, shape):
        return jax.random.normal(k, shape, dtype) * 0.02

    def attn(ks):
        return (w(ks[0], (d, kd)), w(ks[1], (kd,)),
                w(ks[2], (d, kd)), w(ks[3], (kd,)),
                w(ks[4], (d, vd)), w(ks[5], (vd,)),
                w(ks[6], (vd, d)), w(ks[7], (d,)))

    def ln():
        return (jnp.ones((d,), dtype), jnp.zeros((d,), dtype))

    ffn = (w(keys[20], (d, hd)), w(keys[21], (hd,)),
           w(keys[22], (hd, d)), w(keys[23], (d,)))
    return ((dec, enc, source_mask, target_mask)
            + attn(keys[4:12]) + ln()
            + attn(keys[12:20]) + ln()
            + ffn + ln())


def _mha(q_in, kv_in, mask, wq, bq, wk, bk, wv, bv, wo, bo, num_head):
    b, sq, _ = q_in.shape
    sk = kv_in.shape[1]
    # torch uses .view everywhere: head split and the key "transpose" are raw reshapes.
    k = (kv_in @ wk + bk).reshape(b, num_head, -1, sk)
    head_dim = k.shape[2]
    q = (q_in @ wq + bq).reshape(b, num_head, sq, head_dim)
    v = (kv_in @ wv + bv).reshape(b, num_head, sk, -1)
    s = jnp.einsum('abcd,abde->abce', q, k) / (head_dim ** 0.5)
    s = jnp.where(mask == 0, jnp.asarray(CONFIG['mask_fill'], s.dtype), s)
    p = jax.nn.softmax(s, axis=-1)
    o = (p @ v).reshape(b, sq, -1)
    return o @ wo + bo


def _layer_norm(x, g, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['ln_eps']) * g + beta


def workload(dec, enc, source_mask, target_mask,
             sa_wq, sa_bq, sa_wk, sa_bk, sa_wv, sa_bv, sa_wo, sa_bo, ln1_g, ln1_b,
             ca_wq, ca_bq, ca_wk, ca_bk, ca_wv, ca_bv, ca_wo, ca_bo, ln2_g, ln2_b,
             w1, b1, w2, b2, ln3_g, ln3_b):
    num_head = target_mask.shape[1]
    t = _mha(dec, dec, target_mask, sa_wq, sa_bq, sa_wk, sa_bk, sa_wv, sa_bv, sa_wo, sa_bo, num_head)
    t = _layer_norm(t + dec, ln1_g, ln1_b)
    r = t
    t = _mha(t, enc, source_mask, ca_wq, ca_bq, ca_wk, ca_bk, ca_wv, ca_bv, ca_wo, ca_bo, num_head)
    t = _layer_norm(t + r, ln2_g, ln2_b)
    r = t
    t = jax.nn.relu(t @ w1 + b1) @ w2 + b2
    return _layer_norm(t + r, ln3_g, ln3_b)
