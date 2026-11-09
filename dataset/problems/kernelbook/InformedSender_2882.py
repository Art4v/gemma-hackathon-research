"""[bf16 variant] InformedSender: per-game Linear -> conv(game) + sigmoid -> conv(hidden) + sigmoid -> Linear -> scale -> log_softmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'InformedSender_2882',
    'game_size': 16,
    'batch': 1024,
    'feat_size': 2048,
    'embedding_size': 1024,
    'hidden_size': 128,
    'vocab_size': 1024,
    'temp': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['game_size'], c['batch'], c['feat_size']), dtype)
    w1 = jax.random.normal(k1, (c['feat_size'], c['embedding_size']), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['hidden_size'], c['game_size']), dtype) * 0.02
    w3 = jax.random.normal(k3, (c['hidden_size'],), dtype) * 0.02
    w4 = jax.random.normal(k4, (c['embedding_size'], c['vocab_size']), dtype) * 0.02
    return x, w1, w2, w3, w4


def workload(x, w1, w2, w3, w4):
    # The conv kernels span the whole game / hidden axis with matching stride,
    # so each conv is a weighted sum over that axis.
    emb = jnp.einsum('gbf,fe->bge', x, w1)
    h = jax.nn.sigmoid(jnp.einsum('hg,bge->bhe', w2, emb))
    h = jax.nn.sigmoid(jnp.einsum('h,bhe->be', w3, h))
    logits = (h @ w4) * (1.0 / CONFIG['temp'])
    return jax.nn.log_softmax(logits, axis=1)
