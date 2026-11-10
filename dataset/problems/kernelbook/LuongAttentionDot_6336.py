"""[bf16 variant] Luong dot attention: squeeze/unsqueeze/transpose query -> broadcast matmul(values, qT) -> softmax(dim=1) -> weighted sum over dim 1."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LuongAttentionDot_6336',
    'A': 8,    # dim 0 of query/values; the broadcast needs A == B
    'B': 8,
    'C': 512,  # matmul needs C == D
    'D': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['A'], CONFIG['B'], CONFIG['C'], CONFIG['D'])
    kq, kv = jax.random.split(jax.random.key(0))
    query = jax.random.uniform(kq, shape, dtype)
    values = jax.random.uniform(kv, shape, dtype)
    return query, values


def workload(query, values):
    # torch.squeeze(x, 0) only removes dim 0 when its size is 1
    if query.shape[0] == 1:
        query = jnp.squeeze(query, 0)
    query = query[:, None]                        # (A, 1, B, C, D)
    query_t = jnp.swapaxes(query, 2, 1)           # (A, B, 1, C, D)
    # batch dims (A,B) x (A,B,1) broadcast to (A,A,A) as in torch.matmul
    score = jnp.matmul(values, query_t, precision=lax.Precision.HIGHEST)
    attention_weights = jax.nn.softmax(score, axis=1)
    context_vector = jnp.sum(attention_weights * values, axis=1)
    return context_vector, attention_weights
