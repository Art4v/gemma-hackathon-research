"""[bf16 variant] General (Luong) attention: linear_in -> bmm scores -> softmax -> bmm mix -> concat -> linear_out -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_8907',
    'batch': 16,
    'output_len': 1024,
    'query_len': 2048,
    'dimensions': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    b, lq, lk, d = CONFIG['batch'], CONFIG['output_len'], CONFIG['query_len'], CONFIG['dimensions']
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    query = jax.random.uniform(k1, (b, lq, d), dtype)
    context = jax.random.uniform(k2, (b, lk, d), dtype)
    w_in = jax.random.normal(k3, (d, d), dtype) * 0.02
    w_out = jax.random.normal(k4, (2 * d, d), dtype) * 0.02
    return query, context, w_in, w_out


def workload(query, context, w_in, w_out):
    q = jnp.einsum('bld,de->ble', query, w_in)
    scores = jnp.einsum('bqd,bkd->bqk', q, context)
    weights = jax.nn.softmax(scores, axis=-1)
    mix = jnp.einsum('bqk,bkd->bqd', weights, context)
    combined = jnp.concatenate([mix, q], axis=-1)
    output = jnp.tanh(jnp.einsum('bqc,cd->bqd', combined, w_out))
    return output, weights
