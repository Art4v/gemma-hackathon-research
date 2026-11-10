"""MultiHeadedAttention_15199: linear projections of values/keys/queries -> scaled dot-product attention (softmax) -> output linear projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiHeadedAttention_15199',
    'batch': 8,
    'seq': 128,
    'extra': 8,
    'embed_dim': 1024,
    'num_heads': 1,
    'num_projections': 3,
}


def create_inputs(dtype=jnp.float32):
    batch = CONFIG['batch']
    seq = CONFIG['seq']
    extra = CONFIG['extra']
    embed_dim = CONFIG['embed_dim']
    num_projections = CONFIG['num_projections']
    k_values, k_keys, k_queries, k_input, k_output = jax.random.split(jax.random.key(0), 5)
    activation_shape = (batch, seq, extra, embed_dim)
    values = jax.random.uniform(k_values, activation_shape, dtype)
    keys = jax.random.uniform(k_keys, activation_shape, dtype)
    queries = jax.random.uniform(k_queries, activation_shape, dtype)
    input_weights = jax.random.normal(k_input, (embed_dim, num_projections * embed_dim), dtype) * 0.02
    output_weights = jax.random.normal(k_output, (embed_dim, embed_dim), dtype) * 0.02
    return values, keys, queries, input_weights, output_weights


def workload(values, keys, queries, input_weights, output_weights):
    num_heads = CONFIG['num_heads']
    embed_dim = input_weights.shape[0]
    projection_dim = embed_dim // num_heads
    scale = projection_dim ** -0.5
    batch = queries.shape[0]

    def project(x, index):
        start = index * embed_dim
        y = x @ input_weights[:, start:start + embed_dim]
        y = y.reshape(batch, -1, num_heads, projection_dim)
        y = jnp.swapaxes(y, 1, 2)
        return y.reshape(batch * num_heads, -1, projection_dim)

    v = project(values, 0)
    k = project(keys, 1)
    q = project(queries, 2)

    logits = scale * (q @ jnp.swapaxes(k, 1, 2))
    attended = jax.nn.softmax(logits, axis=-1) @ v

    attended = attended.reshape(batch, num_heads, -1, projection_dim)
    attended = jnp.swapaxes(attended, 1, 2)
    attended = attended.reshape(batch, -1, num_heads * projection_dim)
    return attended @ output_weights
