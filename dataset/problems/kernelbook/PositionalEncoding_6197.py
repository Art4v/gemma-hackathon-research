"""[bf16 variant] PositionalEncoding_6197: concat(x, position index) -> Linear(D+1 -> D) -> ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PositionalEncoding_6197',
    'batch': 16,
    'seq_len': 2048,
    'dim_model': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    b, s, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim_model']
    x = jax.random.uniform(kx, (b, s, d), dtype)
    w = jax.random.normal(kw, (d + 1, d), dtype) * 0.02
    bias = jax.random.normal(kb, (d,), dtype) * 0.02
    return x, w, bias


def workload(x, w, bias):
    b, s, _ = x.shape
    pos = jnp.broadcast_to(jnp.arange(s, dtype=x.dtype)[None, :, None], (b, s, 1))
    x_pos = jnp.concatenate([x, pos], axis=2)
    # highest precision: position values up to seq_len-1 are not exact in bf16 passes
    y = jnp.matmul(x_pos, w, precision='highest') + bias
    return jax.nn.relu(y)
