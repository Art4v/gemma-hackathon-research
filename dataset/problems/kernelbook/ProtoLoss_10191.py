"""[bf16 variant] ProtoLoss: support mean -> broadcast squared distance -> log_softmax -> reshape -> cross-entropy and argmax accuracy."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ProtoLoss_10191',
    'num_classes': 128,
    'num_support': 128,
    'num_queries': 128,
    'ndim': 128,
    'label_batch': 8,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kq, kl = jax.random.split(jax.random.key(0), 3)
    c = CONFIG['num_classes']
    s = CONFIG['num_support']
    nq = CONFIG['num_queries']
    d = CONFIG['ndim']
    x = jax.random.uniform(kx, (c, s, d), dtype)
    q = jax.random.uniform(kq, (c, c, d), dtype)
    labels_onehot = jax.random.uniform(kl, (CONFIG['label_batch'], c, nq, c), dtype)
    return x, q, labels_onehot


def workload(x, q, labels_onehot):
    num_classes = q.shape[0]
    ndim = q.shape[2]
    num_queries = labels_onehot.shape[2]

    protox = jnp.mean(1.0 * x.reshape((num_classes, -1, ndim)), axis=1)

    n = protox.shape[0]
    m = q.shape[0]
    a = jnp.repeat(protox[:, None, :], m, axis=1)
    b = jnp.repeat(q[None, ...], n, axis=0)
    dists = 1.0 * jnp.sum(jnp.square(a - b), axis=2)

    logpy = jnp.swapaxes(jax.nn.log_softmax(-1.0 * dists, axis=0), 1, 0).reshape(
        (num_classes, num_queries, num_classes)
    )

    ce_loss = -1.0 * jnp.mean(jnp.mean(logpy * labels_onehot, axis=1))

    float_dtype = labels_onehot.dtype
    label_idx = jnp.argmax(labels_onehot, axis=-1).astype(float_dtype)
    pred_idx = jnp.argmax(logpy, axis=-1).astype(float_dtype)
    accuracy = jnp.mean((label_idx == pred_idx).astype(float_dtype))
    return ce_loss, accuracy
