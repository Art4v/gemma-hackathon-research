"""CrossEntropyLossTF: one_hot(Y) * -log(softmax(Ypred, dim=1)) -> mean * num_classes."""
import jax
import jax.numpy as jnp

# ponytail: the PyTorch module broadcasts the (N, C) one-hot against the trailing (H, W) dims of
# the (N, C, H, W) logits, so height must equal batch and width must equal num_classes.
CONFIG = {
    'name': 'CrossEntropyLossTF_302',
    'batch': 32,
    'num_classes': 256,
    'height': 32,
    'width': 256,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_y = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width'])
    ypred = jax.random.uniform(k_x, shape, dtype)
    y = jax.random.randint(k_y, (CONFIG['batch'],), 0, CONFIG['num_classes'], dtype=jnp.int32)
    return ypred, y


def workload(ypred, y):
    num_classes = ypred.shape[1]
    y_onehot = jax.nn.one_hot(y, num_classes, dtype=jnp.float32)
    logp = jax.nn.log_softmax(ypred, axis=1)
    # Same right-aligned broadcast as torch: (N, C) against (N, C, H, W).
    return jnp.mean(-y_onehot * logp) * num_classes
