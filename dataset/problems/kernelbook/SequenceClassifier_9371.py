"""[bf16 variant] SequenceClassifier: Linear->ReLU->Linear->Sigmoid gate, X*gate, mean(axis=0), Linear->ReLU->Linear->Sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SequenceClassifier_9371',
    'seq': 16,
    'batch': 16,
    'tokens': 256,
    'in_dim': 1024,
    'classes': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    S, B, N = CONFIG['seq'], CONFIG['batch'], CONFIG['tokens']
    D, C = CONFIG['in_dim'], CONFIG['classes']
    ks = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(ks[0], (S, B, N, D), dtype)
    sel_w1 = jax.random.normal(ks[1], (D, D), dtype) * 0.02
    sel_b1 = jax.random.normal(ks[2], (D,), dtype) * 0.02
    sel_w3 = jax.random.normal(ks[3], (D, 1), dtype) * 0.02
    sel_b3 = jax.random.normal(ks[4], (1,), dtype) * 0.02
    pred_w1 = jax.random.normal(ks[5], (D, D), dtype) * 0.02
    pred_b1 = jax.random.normal(ks[6], (D,), dtype) * 0.02
    pred_w3 = jax.random.normal(ks[7], (D, C), dtype) * 0.02
    pred_b3 = jax.random.normal(ks[8], (C,), dtype) * 0.02
    return (x, sel_w1, sel_b1, sel_w3, sel_b3, pred_w1, pred_b1, pred_w3, pred_b3)


def workload(x, sel_w1, sel_b1, sel_w3, sel_b3, pred_w1, pred_b1, pred_w3, pred_b3):
    h = jax.nn.relu(x @ sel_w1 + sel_b1)
    gate = jax.nn.sigmoid(h @ sel_w3 + sel_b3)
    selected = jnp.mean(x * gate, axis=0)
    p = jax.nn.relu(selected @ pred_w1 + pred_b1)
    return jax.nn.sigmoid(p @ pred_w3 + pred_b3)
