"""[bf16 variant] Position embedding lookup (arange over dim 1) plus broadcast add to input embeddings."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PositionEmbedder_11333',
    'batch': 8,
    'seq_len': 256,
    # The [seq_len, embedding_dim] lookup broadcasts against the LAST two
    # input dims, so dim 2 must equal seq_len (mirrors the original 4D input).
    'inner_dim': 256,
    'embedding_dim': 128,
    'max_sequence_length': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['seq_len'], CONFIG['inner_dim'], CONFIG['embedding_dim']),
        dtype,
    )
    table = jax.random.normal(
        k_w, (CONFIG['max_sequence_length'], CONFIG['embedding_dim']), dtype
    ) * 0.02
    return x, table


def workload(input_embeddings, table):
    positions = jnp.arange(input_embeddings.shape[1])
    return input_embeddings + jnp.take(table, positions, axis=0)
