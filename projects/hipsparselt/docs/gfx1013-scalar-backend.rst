Native gfx1013 scalar backend
============================

``HIPSPARSELT_GFX1013_SCALAR_ONLY=ON`` selects a separate HIP implementation for
``GPU_TARGETS=gfx1013``. It does not generate or load Tensile matrix-instruction
kernels. Its purpose is functional structured sparsity on hardware without
sparse matrix instructions; it is not an accelerated sparse tensor-core path.
The default backend and its architecture list remain unchanged.

Implemented operations
----------------------

* FP32 compute with matching FP32, FP16, or BF16 matrix input/output types.
* One structured operand, either A or B; row/column storage and real transpose.
* Strip magnitude pruning and prune checking, including in-place strip pruning.
* Compression and compressed scalar GEMM, through both descriptor and plan APIs.
* Positive, equal batch counts with explicit nonoverlapping padded batch strides.
* Host ``float`` alpha/beta, arbitrary valid leading dimensions, separate C/D
  layouts, and C/D aliasing when their layouts and batch strides match.
* One algorithm (configuration 0), no workspace, and the caller's one HIP stream.
  Search executes this sole algorithm and blocks until its selected/default
  stream completes, without changing its configuration. Only one search
  iteration is accepted. Matmul remains asynchronous.

The fallback rejects tile pruning, integer/FP8/TF32 compute modes, mixed matrix
storage types, bias/activation/vector scaling, split-K, multiple streams, batch
broadcasting, multiple search iterations, and sparse/sparse multiplication. It does not silently ignore
unsupported attributes. Descriptors use bounded positive dimensions to keep
all index and launch products valid.

Compression format
------------------

A compressed buffer is opaque and specific to this backend; it must not be
interchanged with Tensile buffers or persisted across backend changes. For
logical sparse A it stores each row along K; for logical sparse B each column
along K. Batches follow consecutively regardless of the original leading
or batch strides. FP16/BF16 retain two values per four consecutive K elements;
FP32 retains one per two, matching the public sparsity contract. Values occupy
``rows * cols * batches / 2 * sizeof(type)`` bytes. Immediately following is one
metadata byte per group, with ascending, distinct two-bit indices identifying
retained K positions. Unused retained slots hold existing zeros. Compression
requires a valid sparse matrix: prune and/or check first. GEMM decodes this
metadata and never accesses the original dense sparse operand.

Qualification
-------------

``build_tools/bc250/hipsparselt_scalar_smoke.cpp`` in TheRock exercises the public
API against independent CPU magnitude pruning and double-precision dense GEMM,
independently decodes compressed values/metadata, destroys the original sparse
storage before GEMM, checks padding/buffer guards, and detects invalid sparsity.
Its 96 cases combine three types, A/B sparsity, two layouts, four transpose
pairs, and one/two batches. GPU execution must be recorded separately from
host/device compilation; passing syntax checks alone does not qualify runtime
support.
