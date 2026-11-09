"""Similarity: (l*r, |l-r|) concat -> Linear -> sigmoid -> Linear -> log_softmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Similarity_3272',
    'batch': 8192,
    'mem_dim': 2048,
    'hidden_dim': 1024,
    'num_classes': 128,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 6)
    b, m, h, c = CONFIG['batch'], CONFIG['mem_dim'], CONFIG['hidden_dim'], CONFIG['num_classes']
    lvec = jax.random.uniform(k[0], (b, m), dtype)
    rvec = jax.random.uniform(k[1], (b, m), dtype)
    wh_w = jax.random.normal(k[2], (2 * m, h), dtype) * 0.02
    wh_b = jax.random.normal(k[3], (h,), dtype) * 0.02
    wp_w = jax.random.normal(k[4], (h, c), dtype) * 0.02
    wp_b = jax.random.normal(k[5], (c,), dtype) * 0.02
    return lvec, rvec, wh_w, wh_b, wp_w, wp_b


def workload(lvec, rvec, wh_w, wh_b, wp_w, wp_b):
    mult_dist = lvec * rvec
    abs_dist = jnp.abs(lvec - rvec)
    vec_dist = jnp.concatenate([mult_dist, abs_dist], axis=1)
    out = jax.nn.sigmoid(vec_dist @ wh_w + wh_b)
    return jax.nn.log_softmax(out @ wp_w + wp_b, axis=1)
