"""[bf16 variant] FastRCNNPredictor: flatten -> Linear -> ReLU -> Linear -> ReLU -> (Linear cls_score, Linear bbox_pred)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FastRCNNPredictor_4630',
    'batch': 8192,
    'in_channels': 4096,
    'mid_channels': 1024,
    'num_classes': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    cin = CONFIG['in_channels']
    mid = CONFIG['mid_channels']
    nc = CONFIG['num_classes']
    ks = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(ks[0], (b, cin), dtype)
    w1 = jax.random.normal(ks[1], (cin, mid), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (mid,), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (mid, mid), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (mid,), dtype) * 0.02
    wc = jax.random.normal(ks[5], (mid, nc), dtype) * 0.02
    bc = jax.random.normal(ks[6], (nc,), dtype) * 0.02
    wb = jax.random.normal(ks[7], (mid, nc * 4), dtype) * 0.02
    bb = jax.random.normal(ks[8], (nc * 4,), dtype) * 0.02
    return x, w1, b1, w2, b2, wc, bc, wb, bb


def workload(x, w1, b1, w2, b2, wc, bc, wb, bb):
    x = x.reshape(x.shape[0], -1)
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    score = h @ wc + bc
    bbox_delta = h @ wb + bb
    return score, bbox_delta
