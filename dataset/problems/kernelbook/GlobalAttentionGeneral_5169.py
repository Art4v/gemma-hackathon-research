"""[bf16 variant] GlobalAttentionGeneral: conv1x1(context) -> bmm -> softmax over sourceL -> bmm, returning (weightedContext, attn)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GlobalAttentionGeneral_5169',
    'batch': 16,
    'idf': 256,
    'cdf': 256,
    'ih': 64,
    'iw': 64,
    'source_len': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    k_in, k_ctx, k_w = jax.random.split(jax.random.key(0), 3)
    b, idf, cdf = CONFIG['batch'], CONFIG['idf'], CONFIG['cdf']
    ih, iw, s = CONFIG['ih'], CONFIG['iw'], CONFIG['source_len']
    inp = jax.random.uniform(k_in, (b, idf, ih, iw), dtype)
    context = jax.random.uniform(k_ctx, (b, cdf, s), dtype)
    # 1x1 conv weight squeezed to (idf, cdf), no bias.
    w = jax.random.normal(k_w, (idf, cdf), dtype) * 0.02
    return inp, context, w


def workload(inp, context, w):
    b, idf, ih, iw = inp.shape
    s = context.shape[2]
    hp = lax.Precision.HIGHEST
    target = inp.reshape(b, idf, ih * iw)                                     # (B, idf, Q)
    source = jnp.einsum('oc,bcs->bos', w, context, precision=hp)             # (B, idf, S)
    attn = jnp.einsum('bdq,bds->bqs', target, source, precision=hp)          # (B, Q, S)
    # ponytail: mask is None in the module by default, so no masking.
    attn = jax.nn.softmax(attn, axis=-1)
    attn_t = jnp.transpose(attn, (0, 2, 1))                                  # (B, S, Q)
    weighted = jnp.einsum('bds,bsq->bdq', source, attn_t, precision=hp)      # (B, idf, Q)
    return weighted.reshape(b, idf, ih, iw), attn_t.reshape(b, s, ih, iw)
