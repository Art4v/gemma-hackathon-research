"""PKT loss: L2 normalize with NaN scrub, cosine Gram matrices, row-sum normalize, KL-style mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PKT_5786',
    'batch': 4096,
    'features': 2048,
    'eps': 1e-07,
    'similarity_shift': 1.0,
    'similarity_divisor': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_s, k_t = jax.random.split(key)
    shape = (CONFIG['batch'], CONFIG['features'])
    f_s = jax.random.uniform(k_s, shape, dtype)
    f_t = jax.random.uniform(k_t, shape, dtype)
    return f_s, f_t


def workload(f_s, f_t):
    eps = CONFIG['eps']
    shift = CONFIG['similarity_shift']
    divisor = CONFIG['similarity_divisor']

    output_norm = jnp.sqrt(jnp.sum(f_s ** 2, axis=1, keepdims=True))
    output_net = f_s / (output_norm + eps)
    output_net = jnp.where(output_net != output_net, jnp.zeros_like(output_net), output_net)

    target_norm = jnp.sqrt(jnp.sum(f_t ** 2, axis=1, keepdims=True))
    target_net = f_t / (target_norm + eps)
    target_net = jnp.where(target_net != target_net, jnp.zeros_like(target_net), target_net)

    model_similarity = output_net @ output_net.T
    target_similarity = target_net @ target_net.T

    model_similarity = (model_similarity + shift) / divisor
    target_similarity = (target_similarity + shift) / divisor

    model_similarity = model_similarity / jnp.sum(model_similarity, axis=1, keepdims=True)
    target_similarity = target_similarity / jnp.sum(target_similarity, axis=1, keepdims=True)

    return jnp.mean(target_similarity * jnp.log((target_similarity + eps) / (model_similarity + eps)))
