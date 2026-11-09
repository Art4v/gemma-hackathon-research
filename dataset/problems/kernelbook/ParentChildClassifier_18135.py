"""[bf16 variant] ParentChildClassifier: concat(3 inputs) -> Linear -> ReLU -> Linear(2) -> LogSoftmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ParentChildClassifier_18135',
    'batch': 16384,
    'parent_dim': 1024,
    'child_short_dim': 1024,
    'child_full_dim': 1024,
    'hidden_dim': 1024,
    'num_classes': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    b = c['batch']
    in_dim = c['parent_dim'] + c['child_short_dim'] + c['child_full_dim']
    parent = jax.random.uniform(keys[0], (b, c['parent_dim']), dtype)
    child_short = jax.random.uniform(keys[1], (b, c['child_short_dim']), dtype)
    child_full = jax.random.uniform(keys[2], (b, c['child_full_dim']), dtype)
    w_hidden = jax.random.normal(keys[3], (in_dim, c['hidden_dim']), dtype) * 0.02
    b_hidden = jax.random.normal(keys[4], (c['hidden_dim'],), dtype) * 0.02
    w_out = jax.random.normal(keys[5], (c['hidden_dim'], c['num_classes']), dtype) * 0.02
    b_out = jax.random.normal(keys[6], (c['num_classes'],), dtype) * 0.02
    return parent, child_short, child_full, w_hidden, b_hidden, w_out, b_out


def workload(parent, child_short, child_full, w_hidden, b_hidden, w_out, b_out):
    x = jnp.concatenate([parent, child_short, child_full], axis=1)
    h = jax.nn.relu(x @ w_hidden + b_hidden)
    logits = h @ w_out + b_out
    return jax.nn.log_softmax(logits, axis=1)
