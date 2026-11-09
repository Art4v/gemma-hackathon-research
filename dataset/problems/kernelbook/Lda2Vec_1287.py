"""[bf16 variant] Lda2Vec gate_all: softmax(topic_dist) - thr -> matmul topic_emb -> concat -> linear -> sigmoid gate -> blend -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Lda2Vec_1287',
    'batch': 8192,
    'num_topics': 1024,
    'dim': 2048,
    'topic_threshold': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    B, T, D = CONFIG['batch'], CONFIG['num_topics'], CONFIG['dim']
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    word_embedding = jax.random.uniform(k1, (B, D), dtype)
    topic_embedding = jax.random.uniform(k2, (T, D), dtype)
    topic_dist = jax.random.uniform(k3, (B, T), dtype)
    w = jax.random.normal(k4, (2 * D, D), dtype) * 0.02
    b = jax.random.normal(k5, (D,), dtype) * 0.02
    return word_embedding, topic_embedding, topic_dist, w, b


def workload(word_embedding, topic_embedding, topic_dist, w, b):
    td = jax.nn.softmax(topic_dist, axis=1) - CONFIG['topic_threshold']
    topic_hidden = td @ topic_embedding
    context_input = jnp.concatenate((word_embedding, topic_hidden), axis=1)
    gate = jax.nn.sigmoid(context_input @ w + b)
    fusion = gate * word_embedding + (1.0 - gate) * topic_hidden
    return jnp.tanh(fusion)
