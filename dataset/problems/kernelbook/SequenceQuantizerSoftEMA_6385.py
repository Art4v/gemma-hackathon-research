"""SequenceQuantizerSoftEMA: l2-normalize inputs and codebook, softmax similarities, soft reconstruction, L1 + entropy loss."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SequenceQuantizerSoftEMA_6385',
    'batch': 16,
    'seq_len': 4096,
    'd_model': 512,
    'codebook_size': 1024,
    'l1_cost': 1000.0,
    'entropy_cost': 5e-05,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_c = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']), dtype)
    codebook = jax.random.normal(k_c, (CONFIG['codebook_size'], CONFIG['d_model']), dtype) * 0.02
    return x, codebook


def workload(inputs, codebook):
    d_model = codebook.shape[1]
    flat = inputs.reshape(-1, d_model)
    norm_c = codebook / jnp.linalg.norm(codebook, axis=1)[:, None]
    flat = flat / jnp.linalg.norm(flat, axis=1)[:, None]
    distances = jax.nn.softmax(flat @ norm_c.T, axis=1)
    reconstruction = (distances @ norm_c).reshape(inputs.shape)
    l1 = jnp.mean(jnp.abs(distances))
    # ponytail: sum_j (logsm @ d.T)[i, j] == logsm[i] . colsum(d); avoids the N x N matrix.
    logsm = jax.nn.log_softmax(distances, axis=1)
    entropy = jnp.mean(-(logsm @ jnp.sum(distances, axis=0)))
    loss = CONFIG['l1_cost'] * l1 + CONFIG['entropy_cost'] * entropy
    return reconstruction, loss
