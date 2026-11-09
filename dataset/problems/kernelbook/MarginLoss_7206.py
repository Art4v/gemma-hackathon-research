"""[bf16 variant] MarginLoss: capsule L2 norm -> one-hot labels -> relu margin terms -> sum over labels -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MarginLoss_7206',
    'batch': 4096,
    'n_labels': 128,
    'n_features': 128,
    'lambda_': 0.5,
    'm_positive': 0.9,
    'm_negative': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_v, k_l = jax.random.split(jax.random.key(0))
    v = jax.random.uniform(k_v, (CONFIG['batch'], CONFIG['n_labels'], CONFIG['n_features']), dtype)
    labels = jax.random.randint(k_l, (CONFIG['batch'],), 0, CONFIG['n_labels'], dtype=jnp.int32)
    return v, labels


def workload(v, labels):
    v_norm = jnp.sqrt(jnp.sum(v ** 2, axis=-1))
    # n_labels taken from the capsule axis so torch's broadcasting is reproduced for any input rank.
    t = jax.nn.one_hot(labels, v.shape[-2], dtype=v.dtype)
    loss = t * jax.nn.relu(CONFIG['m_positive'] - v_norm) + CONFIG['lambda_'] * (1.0 - t) * jax.nn.relu(
        v_norm - CONFIG['m_negative'])
    return jnp.mean(jnp.sum(loss, axis=-1))
