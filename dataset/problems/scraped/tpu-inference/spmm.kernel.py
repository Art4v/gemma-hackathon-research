"""Pallas TPU N:M structured sparse matmul (from vllm-project/tpu-inference).

Specialized to: rhs_sparse=True, rhs_transpose=True, sparse_dim=1
(weight stored (N, K), 2:4 sparse along the contracting dim with stride 128).
workload(nonzeros, metadata, activation) -> activation @ decompress(W).T
"""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

SPARSITY = (2, 4)
STRIDE = 128
SPARSE_DIM = 1
BLOCK_M, BLOCK_K, BLOCK_N = 256, 512, 512
DEFAULT_VALUE = 0


def next_log2(x):
    return math.ceil(math.log2(x))


def next_pow2(x):
    return 2 ** next_log2(x)


def _get_metadata_packing(sparsity_base):
    md_bitwidth = next_pow2(next_log2(sparsity_base))
    return 32 // md_bitwidth


def _infer_out_dtype(ty1, ty2):
    """Infer matmul output dtype."""
    if ty1 == ty2:
        return ty1
    if jnp.issubdtype(ty1, jnp.integer) == jnp.issubdtype(ty2, jnp.integer):
        return ty1 if jnp.dtype(ty1).itemsize > jnp.dtype(ty2).itemsize else ty2
    return jnp.float32


def _decompress_metadata(md_tile_ref, packing):
    """Decompress metadata. Expected to use in a Pallas kernel."""
    x, h, w = md_tile_ref.shape
    bitwidth = 32 // packing
    decompressed_md = []
    for i in range(h):
        unpacked_md = jnp.broadcast_to(md_tile_ref[:, pl.ds(i, 1), :],
                                       (x, packing, w))
        shift = jax.lax.broadcasted_iota(jnp.int32, unpacked_md.shape,
                                         1) * bitwidth
        unpacked_md = jax.lax.bitwise_and(
            jax.lax.shift_right_logical(unpacked_md, shift),
            jnp.broadcast_to(2**bitwidth - 1, unpacked_md.shape),
        )
        decompressed_md.append(unpacked_md)
    return jnp.concatenate(decompressed_md, axis=1)


def _decompress_nonzeros(sparsity, nonzeros, nonzeros_idx, sparse_dim, stride,
                         default_value):
    """Decompress nonzeros (sparse_dim == 1). Expected to use in a Pallas kernel."""
    x, y = sparsity
    target_size = nonzeros.shape[1 + sparse_dim] * y
    tiles: Any = [None] * (target_size // stride)
    for i in range(y):
        lhs_tile_part = default_value
        for xi in range(x):
            lhs_tile_part = jnp.where(nonzeros_idx[xi] == i, nonzeros[xi],
                                      lhs_tile_part)
        for j in range(target_size // (y * stride)):
            tiles[j * y + i] = lhs_tile_part[:, j * stride:(j + 1) * stride]
    return jnp.concatenate(tiles, axis=sparse_dim).astype(nonzeros.dtype)


@jax.jit
def _structured_spmm(nonzeros, metadata, matrix):
    x, y = SPARSITY
    md_packing = _get_metadata_packing(y)
    m, k = matrix.shape
    n = nonzeros.shape[1]
    block_m, block_k, block_n = BLOCK_M, BLOCK_K, BLOCK_N

    out_dtype = _infer_out_dtype(matrix.dtype, nonzeros.dtype)
    acc_dtype = jnp.float32
    if jnp.issubdtype(out_dtype, jnp.integer):
        acc_dtype = jnp.int32

    def _kernel(nz_tile_ref, md_tile_ref, mat_tile_ref, out_tile_ref,
                acc_tile_ref):

        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_tile_ref[...] = jnp.zeros_like(acc_tile_ref, acc_dtype)

        # Decompress the metadata which contains nonzeros' indices.
        nonzeros_idx = _decompress_metadata(md_tile_ref, md_packing)
        nonzeros_t = nz_tile_ref[...].astype(acc_dtype)
        decompressed_tile = _decompress_nonzeros(
            SPARSITY, nonzeros_t, nonzeros_idx, SPARSE_DIM, STRIDE,
            DEFAULT_VALUE)
        mat_tile = mat_tile_ref[...]
        # rhs_sparse: dense matrix is LHS, decompressed sparse weight is RHS.
        lhs_tile, rhs_tile = mat_tile, decompressed_tile
        packing = 4 // lhs_tile.dtype.itemsize
        if packing > 1 and lhs_tile.shape[0] % (8 * packing) != 0:
            lhs_tile = lhs_tile.astype(acc_dtype)
        if packing > 1 and rhs_tile.shape[0] % (8 * packing) != 0:
            rhs_tile = rhs_tile.astype(acc_dtype)

        # rhs_transpose=True -> contract dim 1 of both.
        dim_nums = (((1,), (1,)), ((), ()))
        acc_tile_ref[...] += jax.lax.dot_general(
            lhs_tile, rhs_tile, dim_nums, preferred_element_type=acc_dtype)

        @pl.when(pl.program_id(2) == k // block_k - 1)
        def _():
            out_tile_ref[...] = acc_tile_ref[...].astype(out_dtype)

    # Sparse operand dims map to (n, k); dense matrix dims map to (m, k).
    nz_shape = (x, block_n, block_k // y)
    md_shape = (x, block_n // md_packing, block_k // y)
    mat_shape = (block_m, block_k)
    in_specs = [
        pl.BlockSpec(nz_shape, lambda i, j, kk: (0, j, kk)),
        pl.BlockSpec(md_shape, lambda i, j, kk: (0, j, kk)),
        pl.BlockSpec(mat_shape, lambda i, j, kk: (i, kk)),
    ]
    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(m // block_m, n // block_n, k // block_k),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, kk: (i, j)),
            scratch_shapes=[pltpu.VMEM((block_m, block_n), acc_dtype)],
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
    )(nonzeros, metadata, matrix)


def workload(nonzeros, metadata, activation):
    return _structured_spmm(nonzeros, metadata, activation)
