"""GlobalAttentionGeneral: flatten, transpose, bmm, softmax over source, transpose, bmm, reshape."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GlobalAttentionGeneral_8369',
    'batch': 8,
    'idf': 128,
    'cdf': 128,
    'content_dim': 128,
    'ih': 128,
    'iw': 128,
    'source_len': 128,
}


def create_inputs(dtype=jnp.float32):
    k_input, k_key, k_value = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_input,
        (CONFIG['batch'], CONFIG['idf'], CONFIG['ih'], CONFIG['iw']),
        dtype,
    )
    context_key = jax.random.uniform(
        k_key,
        (CONFIG['batch'], CONFIG['cdf'], CONFIG['source_len']),
        dtype,
    )
    content_value = jax.random.uniform(
        k_value,
        (CONFIG['batch'], CONFIG['content_dim'], CONFIG['source_len']),
        dtype,
    )
    return x, context_key, content_value


def workload(x, context_key, content_value):
    batch, _, ih, iw = x.shape
    query_len = ih * iw
    target = x.reshape(batch, -1, query_len)
    target_t = jnp.swapaxes(target, 1, 2)
    attn = jnp.matmul(target_t, context_key)
    attn = jax.nn.softmax(attn, axis=-1)
    attn = jnp.swapaxes(attn, 1, 2)
    weighted_context = jnp.matmul(content_value, attn)
    weighted_context = weighted_context.reshape(batch, -1, ih, iw)
    attn = attn.reshape(batch, -1, ih, iw)
    return weighted_context, attn
