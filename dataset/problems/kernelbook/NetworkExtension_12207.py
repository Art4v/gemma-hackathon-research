"""NetworkExtension: x[0], x[1] -> Linear -> sigmoid on both branches (aux branch active since auxiliary and training)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NetworkExtension_12207',
    'num_branches': 2,
    'batch': 64,
    'seq': 512,
    'orig_num_classes': 1024,
    'num_classes': 1024,
    'aux_branch_active': True,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['num_branches'], CONFIG['batch'], CONFIG['seq'], CONFIG['orig_num_classes']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['orig_num_classes'], CONFIG['num_classes']), dtype) * 0.02
    b = jnp.zeros((CONFIG['num_classes'],), dtype)
    return x, w, b


def workload(x, w, b):
    logits = x[0]
    logits_aux = x[1]
    # PyTorch reference runs in training mode with auxiliary truthy, so aux is transformed too.
    if CONFIG['aux_branch_active']:
        logits_aux = jax.nn.sigmoid(logits_aux @ w + b)
    logits = jax.nn.sigmoid(logits @ w + b)
    return logits, logits_aux
