"""[m1 M2_second_output variant of authored/ragged_jagged_softmax/jagged_segment_softmax_3] Jagged segment softmax over packed variable-length segments, also returning the per-segment logsumexp (B, S) of the scaled scores (-inf for empty segments)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'jagged_segment_softmax_m1_aux',
    'batch': 8,
    'total_len': 4194304,
    'num_segments': 4096,
    'head_dim': 128,
    'scale': 1.0 / (128 ** 0.5),
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    T = CONFIG['total_len']
    S = CONFIG['num_segments']
    key = jax.random.key(0)
    k_scores, k_cuts = jax.random.split(key)
    scores = (jax.random.normal(k_scores, (B, T), jnp.float32) * 4.0).astype(dtype)
    cuts = jax.random.randint(k_cuts, (B, S - 1), 0, T + 1, dtype=jnp.int32)
    cuts = jnp.sort(cuts, axis=-1)
    zeros = jnp.zeros((B, 1), jnp.int32)
    ends = jnp.full((B, 1), T, jnp.int32)
    offsets = jnp.concatenate([zeros, cuts, ends], axis=-1)  # (B, S+1)
    return scores, offsets


def workload(scores, offsets):
    B, T = scores.shape
    S = offsets.shape[1] - 1
    scale = CONFIG['scale']

    pos = jnp.arange(T, dtype=jnp.int32)
    # Segment id for every packed position: offsets[s] <= pos < offsets[s+1].
    seg = jax.vmap(lambda o: jnp.searchsorted(o, pos, side='right'))(offsets) - 1
    seg = jnp.clip(seg, 0, S - 1).astype(jnp.int32)
    gid = (seg + jnp.arange(B, dtype=jnp.int32)[:, None] * S).reshape(-1)

    x = scores.astype(jnp.float32).reshape(-1) * scale
    seg_max = jax.ops.segment_max(x, gid, num_segments=B * S,
                                  indices_are_sorted=True)
    e = jnp.exp(x - seg_max[gid])
    seg_sum = jax.ops.segment_sum(e, gid, num_segments=B * S,
                                  indices_are_sorted=True)
    out = e / seg_sum[gid]
    # Per-segment logsumexp (f32), the softmax statistic a backward pass or a segment merge reuses.
    lse = (seg_max + jnp.log(seg_sum)).reshape(B, S)
    return out.reshape(B, T).astype(scores.dtype), lse
