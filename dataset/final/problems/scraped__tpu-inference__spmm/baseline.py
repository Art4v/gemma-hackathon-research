"""N:M structured sparse matmul (SPMM) -- plain JAX reference.

Computes ``activation @ W.T`` where ``W`` (shape (N, K)) is a 2:4 structured
sparse weight stored in compressed form:

  * ``nonzeros``  (x, N, K // y): the x=2 kept values of every group of y=4.
  * ``metadata``  (x, N // 16, K // y) int32: the in-group position (2 bits)
    of each kept value, with 16 consecutive rows packed into one int32 row.

The groups are strided: compressed column ``c = j * stride + s`` expands to the
dense columns ``j * y * stride + i * stride + s`` for ``i in range(y)``.
Masked-out entries take ``default_value`` (0).

This mirrors tpu_inference ``structured_spmm`` with ``rhs_sparse=True``,
``contract_sparse=True``, ``rhs_transpose=True`` (sparse_dim=1).
"""

import jax
import jax.numpy as jnp

CONFIG = dict(
    M=256,            # activation rows
    K=4096,           # contracting dim (sparse dim of the weight)
    N=4096,           # weight output features
    sparsity=(2, 4),  # x:y structured sparsity
    stride=128,       # distance between members of a sparsity group
    block_m=256,
    block_k=512,
    block_n=512,
    md_bitwidth=2,    # bits per packed index = next_pow2(ceil(log2(y)))
    md_packing=16,    # indices per int32 = 32 // md_bitwidth
    default_value=0,
)


def create_inputs(dtype=jnp.float32):
    M, K, N = CONFIG["M"], CONFIG["K"], CONFIG["N"]
    x, y = CONFIG["sparsity"]
    packing, bitwidth = CONFIG["md_packing"], CONFIG["md_bitwidth"]
    k_act, k_nz, k_idx = jax.random.split(jax.random.key(0), 3)

    activation = jax.random.normal(k_act, (M, K), jnp.float32).astype(dtype)
    nonzeros = jax.random.normal(k_nz, (x, N, K // y), jnp.float32).astype(dtype)

    # Random *distinct* positions of the x kept values inside each group of y.
    scores = jax.random.uniform(k_idx, (N, K // y, y))
    idx = jnp.argsort(scores, axis=-1)[..., :x]            # (N, K//y, x)
    idx = jnp.sort(jnp.transpose(idx, (2, 0, 1)), axis=0)  # (x, N, K//y)
    idx = idx.astype(jnp.int32)

    # Pack `packing` consecutive rows of indices into one int32 row.
    shifts = (jnp.arange(packing, dtype=jnp.int32) * bitwidth)[None, None, :, None]
    metadata = jnp.sum(
        idx.reshape(x, N // packing, packing, K // y) << shifts,
        axis=2, dtype=jnp.int32)                            # (x, N//16, K//y)
    return nonzeros, metadata, activation


def _infer_out_dtype(ty1, ty2):
    if ty1 == ty2:
        return ty1
    if jnp.issubdtype(ty1, jnp.integer) == jnp.issubdtype(ty2, jnp.integer):
        return ty1 if jnp.dtype(ty1).itemsize > jnp.dtype(ty2).itemsize else ty2
    return jnp.float32


def workload(nonzeros, metadata, activation):
    x, y = CONFIG["sparsity"]
    stride = CONFIG["stride"]
    packing, bitwidth = CONFIG["md_packing"], CONFIG["md_bitwidth"]
    default_value = CONFIG["default_value"]
    N, Kc = nonzeros.shape[1], nonzeros.shape[2]
    K = Kc * y

    out_dtype = _infer_out_dtype(activation.dtype, nonzeros.dtype)
    acc_dtype = jnp.int32 if jnp.issubdtype(out_dtype, jnp.integer) else jnp.float32

    # Unpack metadata -> per-nonzero in-group index, shape (x, N, K//y).
    shifts = (jnp.arange(packing, dtype=jnp.int32) * bitwidth)[None, None, :, None]
    idx = (metadata[:, :, None, :] >> shifts) & (2 ** bitwidth - 1)
    idx = idx.reshape(x, N, Kc)

    # Rebuild the dense weight: for every in-group slot i, select the nonzero
    # whose index equals i (later nonzeros override), else default_value.
    nz = nonzeros.astype(acc_dtype)
    parts = []
    for i in range(y):
        part = jnp.full((N, Kc), default_value, acc_dtype)
        for xi in range(x):
            part = jnp.where(idx[xi] == i, nz[xi], part)
        parts.append(part)
    dense = jnp.stack(parts, axis=0)                               # (y, N, Kc)
    dense = dense.reshape(y, N, Kc // stride, stride)
    dense = jnp.transpose(dense, (1, 2, 0, 3)).reshape(N, K)       # (N, K)

    out = jax.lax.dot_general(
        activation.astype(acc_dtype), dense,
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=acc_dtype)
    return out.astype(out_dtype)
