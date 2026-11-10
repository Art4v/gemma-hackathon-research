"""[bf16 variant] AdditiveAttention_640: linear -> tanh -> dot query vector -> softmax over candidates -> weighted sum of candidates."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AdditiveAttention_640',
    'batch_size': 128,
    'candidate_size': 512,
    'candidate_vector_dim': 512,
    'query_vector_dim': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b, k_q = jax.random.split(key, 4)
    candidate_vector = jax.random.uniform(
        k_x,
        (CONFIG['batch_size'], CONFIG['candidate_size'], CONFIG['candidate_vector_dim']),
        dtype,
    )
    weight = jax.random.normal(
        k_w, (CONFIG['candidate_vector_dim'], CONFIG['query_vector_dim']), dtype
    ) * 0.02
    bias = jax.random.normal(k_b, (CONFIG['query_vector_dim'],), dtype) * 0.02
    attention_query_vector = jax.random.normal(
        k_q, (CONFIG['query_vector_dim'],), dtype
    ) * 0.02
    return candidate_vector, weight, bias, attention_query_vector


def workload(candidate_vector, weight, bias, attention_query_vector):
    temp = jnp.tanh(candidate_vector @ weight + bias)
    scores = temp @ attention_query_vector
    candidate_weights = jax.nn.softmax(scores, axis=1)
    target = jnp.squeeze(candidate_weights[:, None, :] @ candidate_vector, axis=1)
    return target
