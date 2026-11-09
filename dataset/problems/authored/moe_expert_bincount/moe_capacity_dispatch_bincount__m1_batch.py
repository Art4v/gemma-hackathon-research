"""[m1 M1_batch variant of authored/moe_expert_bincount/moe_capacity_dispatch_bincount] Per-group bincount -> offsets -> stable-rank slots -> capacity mask -> dense combine tensor, now over a leading axis of B independent micro-batches, each holding its own routing groups."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'moe_capacity_dispatch_bincount_m1_batch',
    'num_microbatches': 4,    # independent routing problems
    'num_groups': 72,         # sequences (routing groups) per micro-batch
    'group_size': 4096,       # tokens per group
    'num_experts': 64,
    'top_k': 8,
    'capacity_factor': 1.25,
    'router_bias_scale': 1.5,  # induces expert load imbalance in inputs
}


def _capacity():
    S = CONFIG['group_size']
    K = CONFIG['top_k']
    E = CONFIG['num_experts']
    num = CONFIG['capacity_factor'] * S * K
    return int(-(-num // E))  # ceil


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['num_microbatches']
    G = CONFIG['num_groups']
    S = CONFIG['group_size']
    E = CONFIG['num_experts']
    K = CONFIG['top_k']
    T = G * S
    key = jax.random.key(0)
    k_logits, = jax.random.split(key, 1)
    logits = jax.random.normal(k_logits, (B, T, E), jnp.float32)
    bias = jnp.linspace(-CONFIG['router_bias_scale'], CONFIG['router_bias_scale'], E,
                        dtype=jnp.float32)
    logits = logits + bias[None, None, :]
    top_vals, top_idx = lax.top_k(logits, K)
    gates = jax.nn.softmax(top_vals, axis=-1).astype(dtype)
    return top_idx.astype(jnp.int32), gates


def _route_microbatch(expert_idx, gates):
    G = CONFIG['num_groups']
    S = CONFIG['group_size']
    E = CONFIG['num_experts']
    K = expert_idx.shape[-1]
    C = _capacity()
    N = G * K * S

    idx = expert_idx.reshape(G, S, K)
    gts = gates.reshape(G, S, K)

    # Priority order within a group: choice-major (all top-1 before top-2, ...), then token order.
    e_flat = jnp.transpose(idx, (0, 2, 1)).reshape(-1)                      # (G*K*S,)
    g_flat = jnp.repeat(jnp.arange(G, dtype=jnp.int32), K * S)              # (G*K*S,)
    seg = g_flat * E + e_flat                                               # segment key

    # Per-group expert histogram and offsets.
    counts_flat = jnp.bincount(seg, length=G * E).astype(jnp.int32)         # (G*E,)
    counts = counts_flat.reshape(G, E)
    offsets = (jnp.cumsum(counts, axis=1) - counts).astype(jnp.int32)       # within-group exclusive
    seg_start = (jnp.cumsum(counts_flat) - counts_flat).astype(jnp.int32)   # global exclusive

    # Position of each assignment inside its expert buffer via stable sort rank.
    order = jnp.argsort(seg, stable=True)
    rank = jnp.zeros((N,), jnp.int32).at[order].set(jnp.arange(N, dtype=jnp.int32))
    pos_flat = rank - seg_start[seg]

    pos = jnp.transpose(pos_flat.reshape(G, K, S), (0, 2, 1))               # (G, S, K)
    kept = pos < C

    slot = jnp.where(kept, idx * C + pos, -1).astype(jnp.int32)             # (G, S, K)

    # Dense gate-scaled dispatch/combine tensor, accumulated in float32.
    w = jnp.where(kept, gts.astype(jnp.float32), 0.0)
    gi = jnp.arange(G, dtype=jnp.int32)[:, None, None]
    si = jnp.arange(S, dtype=jnp.int32)[None, :, None]
    combine = jnp.zeros((G, S, E), jnp.float32).at[gi, si, idx].add(w)
    combine = combine.astype(gates.dtype)

    return counts, offsets, slot.reshape(G * S, K), combine


def workload(expert_idx, gates):
    return jax.vmap(_route_microbatch)(expert_idx, gates)
