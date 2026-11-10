"""TopicEmbeddingAttention: W @ topic_emb.T -> encoder_memory @ (.) -> softmax(axis=2) -> (.) @ topic_emb."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TopicEmbeddingAttention_1243',
    'encoder_memory_shape': (64, 64, 128, 128),
    'topic_emb_shape': (64, 64, 64, 64),
    'encoder_hidden_size': 128,
    'topic_num': 64,
    'topic_emb_dim': 64,
    'softmax_axis': 2,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_memory, k_topic, k_w = jax.random.split(key, 3)
    encoder_memory = jax.random.uniform(k_memory, CONFIG['encoder_memory_shape'], dtype)
    topic_emb = jax.random.uniform(k_topic, CONFIG['topic_emb_shape'], dtype)
    w_shape = (CONFIG['encoder_hidden_size'], CONFIG['topic_emb_dim'])
    w = jax.random.normal(k_w, w_shape, dtype) * CONFIG['weight_scale']
    return encoder_memory, topic_emb, w


def workload(encoder_memory, topic_emb, w):
    topic_seq_w = jnp.matmul(w, jnp.transpose(topic_emb))
    seq_topic_w = jnp.matmul(encoder_memory, topic_seq_w)
    seq_topic_w = jax.nn.softmax(seq_topic_w, axis=CONFIG['softmax_axis'])
    hidden_topic_state = jnp.matmul(seq_topic_w, topic_emb)
    return hidden_topic_state
