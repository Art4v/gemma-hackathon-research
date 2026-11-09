"""RewardCriterion: gather(log-probs, seq) -> shifted (seq>0) mask -> -logp*reward*mask -> masked mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RewardCriterion_11442',
    'batch': 128,
    'seq_len': 128,
    'vocab': 4096,
    'reduction': 'mean',
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    n, l, v = CONFIG['batch'], CONFIG['seq_len'], CONFIG['vocab']
    logprobs = jax.random.uniform(k1, (n, l, v), dtype)
    seq = jax.random.randint(k2, (n, l), 0, v, dtype=jnp.int32)
    reward = jax.random.uniform(k3, (n, l), dtype)
    return logprobs, seq, reward


def workload(logprobs, seq, reward):
    n = seq.shape[0]
    gathered = jnp.take_along_axis(logprobs, seq[:, :, None], axis=2)[:, :, 0]
    mask = jnp.concatenate(
        [jnp.ones((n, 1), dtype=bool), (seq > 0)[:, :-1]], axis=1
    ).astype(gathered.dtype)
    output = -gathered * reward * mask
    if CONFIG['reduction'] == 'none':
        return output.sum(1) / mask.sum(1)
    return jnp.sum(output) / jnp.sum(mask)
