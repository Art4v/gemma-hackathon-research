"""[bf16 variant] Split cross-entropy loss (log_q branch): linear -> logsumexp -> q-log transform -> gather -> mean NLL."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SplitCrossEntropyLoss_9282',
    'n_tokens': 4096,
    'hidden_size': 1024,
    'vocab_size': 16384,
    'q': 4.0,
    'b': 4.0,
    'g': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    n, h, v = CONFIG['n_tokens'], CONFIG['hidden_size'], CONFIG['vocab_size']
    hiddens = jax.random.uniform(k1, (n, h), dtype)
    targets = jax.random.randint(k2, (n,), 0, v, dtype=jnp.int32)
    weight = jax.random.normal(k3, (h, v), dtype) * 0.02
    bias = jax.random.normal(k4, (v,), dtype) * 0.02
    return hiddens, targets, weight, bias


def _logsumexp(x):
    xm = jnp.max(x, axis=-1, keepdims=True)
    lse = xm + jnp.log(jnp.sum(jnp.exp(x - xm), axis=-1, keepdims=True))
    return jnp.where(jnp.isinf(xm), xm, lse)


def workload(hiddens, targets, weight, bias):
    # ponytail: training=True and q != 1 always selects log_q; b/g branches are unreachable here.
    q = CONFIG['q']
    hiddens = hiddens.reshape(-1, hiddens.shape[-1])
    logits = hiddens @ weight + bias
    lnorm = _logsumexp(logits)
    picked = jnp.take_along_axis(logits, targets.reshape(-1, 1), axis=1)
    log_q = jnp.expm1((1.0 - q) * (picked - lnorm)) / ((1.0 - q) * q)
    total_loss = -jnp.sum(log_q.astype(jnp.float32))
    return (total_loss / targets.shape[0]).astype(weight.dtype)
