Native gfx1013 scalar GEMM
=========================

``HIPBLASLT_ENABLE_GFX1013_SCALAR=ON`` builds a native HIP implementation of
``hipblasLtMatmul`` and ``hipblasLtMatmulAlgoGetHeuristic`` for gfx1013. The
option defaults to OFF and requires ``GPU_TARGETS=gfx1013`` and
``HIPBLASLT_HOST_API_ONLY=ON``. The latter selects the descriptor library
without Tensile device libraries; the scalar option adds its own GPU backend.
Device-library, rocRoller, ExtOp and MatrixTransform options must remain OFF.
Other architecture builds retain their existing dispatch.

Supported operations
--------------------

* FP32, FP16 or BF16 A and B, with matching input types and FP32 accumulation.
* C and D both use the input type or FP32.
* Column-major and row-major layouts, with padded leading dimensions.
* Independent transpose or non-transpose operations on A and B.
* Strided batches with matching batch counts; zero input batch stride broadcasts
  a matrix across batches. Output batches must not overlap.
* Host or device FP32 alpha and beta. Host beta=0 permits a null C pointer;
  device scalar mode requires C because beta is not inspected on the host.
* In-place C/D with identical layout, stride and offset, or separate C/D with
  different leading dimensions. A zero beta does not read C. A zero alpha does
  not read A or B. K=0 scales C without reading A or B.
* Explicit heuristic algorithms or a null algorithm for implicit selection.
  The scalar kernel needs no workspace and executes on the supplied HIP stream.
* Bias, ReLU, GELU, fixed-beta-one Swish, sigmoid, clamp and applicable bias
  combinations. Clamp uses the supplied activation bounds. GELU/clamp auxiliary
  outputs and DGELU are supported with column-major auxiliary storage.
* Native C++ Gemm/GroupedGemm host-problem APIs, owning descriptor clones and
  host scalar snapshots, heuristic/support/index queries and stream/event execution.
* Matching FP16/BF16 FAST compute aliases use FP32 scalar accumulation.
* Bias, ReLU, GELU and their bias combinations. Bias follows logical output rows;
  output-type or FP32 bias supports broadcast or padded batch strides.

Limitations
-----------

This is a correctness-oriented scalar FMA implementation using 16x16 shared
memory tiles. It does not claim performance parity with optimized BLAS kernels.
Matrix instructions are not emitted because gfx1013 has neither MFMA nor WMMA.

Bias-gradient and packed-ReLU auxiliary epilogues, scaling pointers, FP8/INT8/FP64,
pointer-array batches, device user-argument grouped execution, packed layouts and
C leading-dimension-zero broadcasting are unsupported. Non-default SM-count,
Stream-K and search/configuration controls are rejected rather than ignored.
Numerics-scanner requests are also unsupported. The handle must belong to the
current gfx1013 device. Matrix dimensions and leading dimensions are bounded by
INT32_MAX, and address spans and grid size are checked before launch.

The hardware probe ``build_tools/bc250/hipblaslt_scalar_smoke.cpp`` in TheRock
checks public API dispatch against an independent CPU double-precision reference,
including padding, transposes, batches, beta=0, K=0, alpha=0 and rejection cases.
