"""[bf16 variant] MeanMaxPooling: mask-broadcast doc states per entity, then concat max-pool and length-normalized sum-pool."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MeanMaxPooling_8189',
    'batch': 8,
    'doc_len': 8192,
    'num_entities': 128,
    'hidden': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    n = CONFIG['batch']
    l = CONFIG['doc_len']
    e = CONFIG['num_entities']
    d = CONFIG['hidden']
    k_doc, k_map, k_lens = jax.random.split(jax.random.key(0), 3)
    doc_state = jax.random.uniform(k_doc, (n, l, d), dtype)
    entity_mapping = jax.random.uniform(k_map, (n, e, l), dtype)
    entity_lens = jax.random.randint(k_lens, (n, e), 1, l + 1).astype(dtype)
    return doc_state, entity_mapping, entity_lens


def workload(doc_state, entity_mapping, entity_lens):
    entity_states = jnp.expand_dims(entity_mapping, 3) * jnp.expand_dims(doc_state, 1)
    max_pooled = jnp.max(entity_states, axis=2)
    mean_pooled = jnp.sum(entity_states, axis=2) / jnp.expand_dims(entity_lens, 2)
    return jnp.concatenate([max_pooled, mean_pooled], axis=2)
