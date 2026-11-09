"""Attention_17819: input_proj -> bmm scores -> softmax -> bmm mix -> concat(mix, input) -> output_proj -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_17819',
    'batch': 16,
    'tgt_len': 1024,
    'src_len': 1024,
    'input_dim': 1024,
    'source_dim': 1024,
    'output_dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_in, k_src, k_wi, k_wo = jax.random.split(jax.random.key(0), 4)
    b, t, s = CONFIG['batch'], CONFIG['tgt_len'], CONFIG['src_len']
    d_in, d_src, d_out = CONFIG['input_dim'], CONFIG['source_dim'], CONFIG['output_dim']
    inp = jax.random.uniform(k_in, (b, t, d_in), dtype)
    src = jax.random.uniform(k_src, (b, s, d_src), dtype)
    w_in = jax.random.normal(k_wi, (d_in, d_src), dtype) * 0.02
    w_out = jax.random.normal(k_wo, (d_in + d_src, d_out), dtype) * 0.02
    return inp, src, w_in, w_out


def workload(inp, src, w_in, w_out):
    x = inp @ w_in
    attn = jax.nn.softmax(jnp.einsum('btd,bsd->bts', x, src), axis=-1)
    mix = jnp.einsum('bts,bsd->btd', attn, src)
    combined = jnp.concatenate([mix, inp], axis=-1)
    output = jnp.tanh(combined @ w_out)
    return output, attn
