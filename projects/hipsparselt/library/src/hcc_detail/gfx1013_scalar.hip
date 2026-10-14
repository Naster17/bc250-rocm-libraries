// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Native gfx1013 fallback: ordinary scalar HIP instructions, host FP32 scalars,
// strip/4x4 tile pruning, and scalar activation epilogues. The compressed buffer is backend-specific:
// tightly packed retained values for each logical contraction row followed by
// one byte per group containing two-bit indices. FP16/BF16 use 2:4, FP32 1:2.
// Its orientation follows sparse A's rows or sparse B's columns after op().
// Compression never aliases a dense buffer and GEMM reads only these values
// and their indices, not the original dense matrix.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipsparselt/hipsparselt.h>
#include <limits>
#include <type_traits>

#define HIPSPARSELT_STRINGIFY_IMPL(value) #value
#define HIPSPARSELT_STRINGIFY(value) HIPSPARSELT_STRINGIFY_IMPL(value)

namespace
{
    constexpr uint64_t handle_tag = 0x10130001, matrix_tag = 0x10130002, multiply_tag = 0x10130003,
                       algorithm_tag = 0x10130004, plan_tag = 0x10130005;
    struct Handle
    {
        uint64_t tag    = handle_tag;
        int      device = 0;
    };
    struct Matrix
    {
        uint64_t         tag = matrix_tag;
        int64_t          rows, cols, ld, batch_stride;
        int              batches = 1;
        hipDataType      type;
        hipsparseOrder_t order;
        bool             sparse;
    };
    struct Multiply
    {
        uint64_t             tag = multiply_tag;
        Matrix               a, b, c, d;
        hipsparseOperation_t opa, opb;
        int                  activation = -1;
        float                leakyAlpha = 0.01f;
        float       reluThreshold = 0.0f, reluUpper = std::numeric_limits<float>::infinity();
        float       geluScale = 1.0f, tanhAlpha = 1.0f, tanhBeta = 1.0f;
        const void* bias         = nullptr;
        int64_t     bias_stride  = 0;
        hipDataType bias_type    = HIP_R_32F;
        int         alpha_vector = 0;
    };
    struct Algorithm
    {
        uint64_t tag        = algorithm_tag;
        int      iterations = 1;
    };
    struct Plan
    {
        uint64_t tag = plan_tag;
        Multiply mm;
    };
    bool activationAttribute(hipsparseLtMatmulDescAttribute_t attr)
    {
        return attr == HIPSPARSELT_MATMUL_ACTIVATION_RELU
               || attr == HIPSPARSELT_MATMUL_ACTIVATION_ABS
               || attr == HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU
               || attr == HIPSPARSELT_MATMUL_ACTIVATION_GELU
               || attr == HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID
               || attr == HIPSPARSELT_MATMUL_ACTIVATION_TANH;
    }
    float* activationParameter(Multiply& mm, hipsparseLtMatmulDescAttribute_t attr)
    {
        switch(attr)
        {
        case HIPSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD:
            return &mm.reluThreshold;
        case HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND:
            return &mm.reluUpper;
        case HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU_ALPHA:
            return &mm.leakyAlpha;
        case HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING:
            return &mm.geluScale;
        case HIPSPARSELT_MATMUL_ACTIVATION_TANH_ALPHA:
            return &mm.tanhAlpha;
        case HIPSPARSELT_MATMUL_ACTIVATION_TANH_BETA:
            return &mm.tanhBeta;
        default:
            return nullptr;
        }
    }
    static_assert(sizeof(Plan) <= sizeof(hipsparseLtMatmulPlan_t));
    static_assert(sizeof(Multiply) <= sizeof(hipsparseLtMatmulDescriptor_t));
    template <class T, class O>
    bool read(const O* opaque, T& value, uint64_t tag)
    {
        if(!opaque)
            return false;
        std::memcpy(&value, opaque->data, sizeof(T));
        return value.tag == tag;
    }
    template <class T, class O>
    void write(O* opaque, const T& value)
    {
        std::memset(opaque->data, 0, sizeof(opaque->data));
        std::memcpy(opaque->data, &value, sizeof(T));
    }
    template <class O>
    hipsparseStatus_t destroy(const O* object, uint64_t tag)
    {
        uint64_t actual = 0;
        if(object)
            std::memcpy(&actual, object->data, sizeof(actual));
        if(actual != tag)
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memset(const_cast<O*>(object)->data, 0, sizeof(object->data));
        return HIPSPARSE_STATUS_SUCCESS;
    }
    bool valid(const hipsparseLtHandle_t* handle)
    {
        Handle h;
        int    device;
        return read(handle, h, handle_tag) && hipGetDevice(&device) == hipSuccess
               && device == h.device;
    }
    hipsparseStatus_t hip_status(hipError_t error)
    {
        if(error != hipSuccess && std::getenv("HIPSPARSELT_SCALAR_DIAGNOSTICS"))
            std::fprintf(stderr,
                         "hipSPARSELt scalar HIP error: %s (%s)\n",
                         hipGetErrorName(error),
                         hipGetErrorString(error));
        return error == hipSuccess ? HIPSPARSE_STATUS_SUCCESS : HIPSPARSE_STATUS_EXECUTION_FAILED;
    }
    __host__ __device__ int64_t rows(const Matrix& a, hipsparseOperation_t op)
    {
        return op == HIPSPARSE_OPERATION_NON_TRANSPOSE ? a.rows : a.cols;
    }
    __host__ __device__ int64_t cols(const Matrix& a, hipsparseOperation_t op)
    {
        return op == HIPSPARSE_OPERATION_NON_TRANSPOSE ? a.cols : a.rows;
    }
    __host__ __device__ int64_t
        offset(const Matrix& a, hipsparseOperation_t op, int64_t row, int64_t col, int batch)
    {
        if(op != HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            auto tmp = row;
            row      = col;
            col      = tmp;
        }
        return (a.batches == 1 ? 0 : batch * a.batch_stride)
               + (a.order == HIPSPARSE_ORDER_ROW ? row * a.ld + col : col * a.ld + row);
    }
    bool operation(hipsparseOperation_t op)
    {
        return op == HIPSPARSE_OPERATION_NON_TRANSPOSE || op == HIPSPARSE_OPERATION_TRANSPOSE;
    }
    size_t width(hipDataType type)
    {
        return type == HIP_R_8I ? 1 : (type == HIP_R_32F ? 4 : 2);
    }
    int group_size(const Matrix& matrix)
    {
        return matrix.type == HIP_R_32F ? 2 : 4;
    }
    int64_t outer(const Matrix& matrix, bool sparse_a, hipsparseOperation_t op)
    {
        return sparse_a ? rows(matrix, op) : cols(matrix, op);
    }
    __host__ __device__ int64_t contraction(const Matrix&        matrix,
                                            bool                 sparse_a,
                                            hipsparseOperation_t op)
    {
        return sparse_a ? cols(matrix, op) : rows(matrix, op);
    }
    const Matrix& sparse(const Multiply& mm)
    {
        return mm.a.sparse ? mm.a : mm.b;
    }
    hipsparseOperation_t sparse_op(const Multiply& mm)
    {
        return mm.a.sparse ? mm.opa : mm.opb;
    }
    int64_t groups(const Matrix& matrix)
    {
        return matrix.rows * matrix.cols / group_size(matrix);
    }
    size_t value_bytes(const Matrix& matrix)
    {
        return matrix.rows * matrix.cols / 2 * (matrix.batch_stride == 0 ? 1 : matrix.batches)
               * width(matrix.type);
    }
    size_t compressed_bytes(const Matrix& matrix)
    {
        return value_bytes(matrix)
               + groups(matrix) * (matrix.batch_stride == 0 ? 1 : matrix.batches);
    }

    size_t dense_bytes(const Matrix& matrix)
    {
        size_t span = matrix.ld * (matrix.order == HIPSPARSE_ORDER_ROW ? matrix.rows : matrix.cols);
        return (span + (matrix.batches - 1) * matrix.batch_stride) * width(matrix.type);
    }
    bool overlaps(const void* a, size_t a_bytes, const void* b, size_t b_bytes)
    {
        auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
        return x < y ? y - x < a_bytes : x - y < b_bytes;
    }
    size_t bias_bytes(const Multiply& mm)
    {
        return (mm.d.rows + (mm.d.batches - 1) * mm.bias_stride) * width(mm.bias_type);
    }

    // A group owns all its dense output locations, allowing in-place strip pruning.
    template <class T, bool Check, bool Compress>
    __global__ void strip_kernel(Matrix               matrix,
                                 bool                 sparse_a,
                                 hipsparseOperation_t op,
                                 const T*             input,
                                 T*                   output,
                                 int*                 invalid)
    {
        int64_t group = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        int     size = matrix.type == HIP_R_32F ? 2 : 4, kept = size / 2;
        int64_t k = contraction(matrix, sparse_a, op), per_row = k / size;
        int64_t per_batch = matrix.rows * matrix.cols / size;
        if(group >= per_batch * matrix.batches)
            return;
        int     batch   = group / per_batch;
        int64_t logical = group % per_batch, row = logical / per_row,
                start = logical % per_row * size;
        T       values[4];
        int     indices[4] = {0, 1, 2, 3};
        int     nonzero    = 0;
        for(int i = 0; i < size; ++i)
        {
            values[i] = input[offset(
                matrix, op, sparse_a ? row : start + i, sparse_a ? start + i : row, batch)];
            nonzero += float(values[i]) != 0;
        }
        if constexpr(Check)
        {
            if(nonzero > kept)
                atomicExch(invalid, 1);
            return;
        }
        // Descending absolute value, stable lowest-index tie break. NaNs retained.
        for(int i = 0; i < size; ++i)
            for(int j = i + 1; j < size; ++j)
            {
                float a = fabsf(float(values[indices[i]])), b = fabsf(float(values[indices[j]]));
                if((isnan(b) && !isnan(a)) || b > a
                   || ((b == a || (isnan(a) && isnan(b))) && indices[j] < indices[i]))
                {
                    int tmp    = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp;
                }
            }
        if constexpr(Compress)
        {
            // Compression selects existing nonzeros (not magnitude pruning). Input
            // must satisfy PruneCheck; fill unused slots with distinct zero indices.
            int selected[2] = {0, 0}, count = 0;
            for(int i = 0; i < size && count < kept; ++i)
                if(float(values[i]) != 0)
                    selected[count++] = i;
            for(int i = 0; i < size && count < kept; ++i)
                if(float(values[i]) == 0)
                    selected[count++] = i;
            if(kept == 2 && selected[0] > selected[1])
            {
                int tmp     = selected[0];
                selected[0] = selected[1];
                selected[1] = tmp;
            }
            unsigned char md = 0;
            for(int i = 0; i < kept; ++i)
            {
                output[group * kept + i] = values[selected[i]];
                md |= selected[i] << (2 * i);
            }
            auto metadata   = reinterpret_cast<unsigned char*>(output)
                              + matrix.rows * matrix.cols / 2 * matrix.batches * sizeof(T);
            metadata[group] = md;
        }
        else
        {
            for(int i = 0; i < size; ++i)
            {
                bool retain = false;
                for(int j = 0; j < kept; ++j)
                    retain |= indices[j] == i;
                output[offset(
                    matrix, op, sparse_a ? row : start + i, sparse_a ? start + i : row, batch)]
                    = retain ? values[i] : T(0.0f);
            }
        }
    }

    template <class T>
    __global__ void tile_kernel(
        Matrix matrix, bool sparse_a, hipsparseOperation_t op, const T* input, T* output)
    {
        int64_t tile    = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        int64_t k       = contraction(matrix, sparse_a, op);
        int64_t length  = sparse_a ? rows(matrix, op) : cols(matrix, op);
        int64_t columns = k / 4, per_batch = ((length + 3) / 4) * columns;
        if(tile >= per_batch * matrix.batches)
            return;
        int     batch     = tile / per_batch;
        int64_t first_row = ((tile % per_batch) / columns) * 4;
        int64_t first_col = (tile % columns) * 4;
        T       values[16];
        double  magnitude[16];
        for(int r = 0; r < 4; ++r)
            for(int c = 0; c < 4; ++c)
            {
                values[r * 4 + c]    = first_row + r < length
                                           ? input[offset(matrix,
                                                          op,
                                                          sparse_a ? first_row + r : first_col + c,
                                                          sparse_a ? first_col + c : first_row + r,
                                                          batch)]
                                           : T(0.0f);
                magnitude[r * 4 + c] = fabs(double(float(values[r * 4 + c])));
            }
        // Enumerate the six 2-of-4 masks for each of the first three rows.
        // Column counts uniquely determine the fourth row. Precisely 90 of
        // these 216 candidates have two selections in every row AND column.
        // FP64 scores avoid overflowing finite BF16 magnitudes in FP32.
        constexpr unsigned masks[6]   = {3, 5, 6, 9, 10, 12};
        unsigned           best_mask  = 0xCC33;
        double             best_score = -1;
        for(unsigned a : masks)
            for(unsigned b : masks)
                for(unsigned c : masks)
                {
                    unsigned d          = 0;
                    bool     admissible = true;
                    for(int column = 0; column < 4; ++column)
                    {
                        int count = ((a >> column) & 1) + ((b >> column) & 1) + ((c >> column) & 1);
                        admissible &= count == 1 || count == 2;
                        if(count == 1)
                            d |= 1u << column;
                    }
                    if(!admissible || __builtin_popcount(d) != 2)
                        continue;
                    unsigned mask  = a | (b << 4) | (c << 8) | (d << 12);
                    double   score = 0;
                    for(int i = 0; i < 16; ++i)
                        if((mask >> i) & 1)
                            score += magnitude[i];
                    if(score > best_score || (score == best_score && mask < best_mask))
                    {
                        best_score = score;
                        best_mask  = mask;
                    }
                }
        for(int r = 0; r < 4 && first_row + r < length; ++r)
            for(int c = 0; c < 4; ++c)
                output[offset(matrix,
                              op,
                              sparse_a ? first_row + r : first_col + c,
                              sparse_a ? first_col + c : first_row + r,
                              batch)]
                    = ((best_mask >> (r * 4 + c)) & 1) ? values[r * 4 + c] : T(0.0f);
    }

    hipsparseStatus_t launch_tile(const hipsparseLtHandle_t* handle,
                                  Matrix                     matrix,
                                  int                        sparse_a,
                                  hipsparseOperation_t       op,
                                  const void*                input,
                                  void*                      output,
                                  hipStream_t                stream)
    {
        if(!valid(handle) || !matrix.sparse || !operation(op) || !input || !output
           || (sparse_a != 0 && sparse_a != 1))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        // TILE's 2-per-row/column masks are incompatible with FP32's 1:2
        // contract: two retained values can land in the same adjacent pair.
        if(matrix.type != HIP_R_16F && matrix.type != HIP_R_16BF && matrix.type != HIP_R_8I)
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        int64_t k = contraction(matrix, sparse_a, op);
        if(k % 4)
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        if(input != output && overlaps(input, dense_bytes(matrix), output, dense_bytes(matrix)))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        // A zero-stride input is one physical matrix. Each tile/group must have
        // a single writer even when the descriptor has several logical batches.
        if(matrix.batch_stride == 0)
            matrix.batches = 1;
        int64_t length = sparse_a ? rows(matrix, op) : cols(matrix, op);
        int64_t count  = ((length + 3) / 4) * (k / 4) * matrix.batches;
        dim3    grid((count + 63) / 64), block(64);
        if(matrix.type == HIP_R_16F)
            hipLaunchKernelGGL(tile_kernel<__half>,
                               grid,
                               block,
                               0,
                               stream,
                               matrix,
                               bool(sparse_a),
                               op,
                               (const __half*)input,
                               (__half*)output);
        else if(matrix.type == HIP_R_8I)
            hipLaunchKernelGGL(tile_kernel<int8_t>,
                               grid,
                               block,
                               0,
                               stream,
                               matrix,
                               bool(sparse_a),
                               op,
                               (const int8_t*)input,
                               (int8_t*)output);
        else
            hipLaunchKernelGGL(tile_kernel<hip_bfloat16>,
                               grid,
                               block,
                               0,
                               stream,
                               matrix,
                               bool(sparse_a),
                               op,
                               (const hip_bfloat16*)input,
                               (hip_bfloat16*)output);
        return hip_status(hipGetLastError());
    }

    template <class T>
    __global__ void multiply_kernel(Multiply     mm,
                                    float        alpha,
                                    float        beta,
                                    const float* alpha_vec,
                                    const T*     a,
                                    const T*     b,
                                    const T*     c,
                                    T*           d)
    {
        int64_t m = rows(mm.a, mm.opa), n = cols(mm.b, mm.opb), k = cols(mm.a, mm.opa);
        int64_t element = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if(element >= m * n * mm.d.batches)
            return;
        int     batch = element / (m * n);
        int64_t row = (element % (m * n)) / n, col = element % n;
        if(mm.alpha_vector)
            alpha = alpha_vec[row];
        Matrix   sm         = mm.a.sparse ? mm.a : mm.b;
        const T* compressed = mm.a.sparse ? a : b;
        int      size = sm.type == HIP_R_32F ? 2 : 4, kept = size / 2;
        int64_t  group_count = sm.rows * sm.cols / size;
        auto     metadata
            = reinterpret_cast<const unsigned char*>(compressed)
              + sm.rows * sm.cols / 2 * (sm.batch_stride == 0 ? 1 : sm.batches) * sizeof(T);
        int64_t base = (sm.batches == 1 || sm.batch_stride == 0 ? 0 : batch * group_count)
                       + (mm.a.sparse ? row : col) * (k / size);
        // At most K/2 retained INT8 products. Descriptor K bound proves
        // every partial sum fits int32; convert once for the FP32 epilogue.
        using Accumulator = std::conditional_t<std::is_same_v<T, int8_t>, int32_t, float>;
        Accumulator sum   = 0;
        for(int64_t g = 0; alpha != 0 && g < k / size; ++g)
            for(int t = 0; t < kept; ++t)
            {
                int64_t     kk = g * size + ((metadata[base + g] >> (t * 2)) & 3);
                Accumulator sv = Accumulator(compressed[(base + g) * kept + t]);
                Accumulator dv = mm.a.sparse ? Accumulator(b[offset(mm.b, mm.opb, kk, col, batch)])
                                             : Accumulator(a[offset(mm.a, mm.opa, row, kk, batch)]);
                sum += sv * dv;
            }
        // beta==0 must not read C, which may contain NaNs or uninitialized data.
        float value = alpha * float(sum);
        if(beta != 0)
            value += beta
                     * float(c[offset(mm.c, HIPSPARSE_OPERATION_NON_TRANSPOSE, row, col, batch)]);
        if(mm.bias)
        {
            int64_t index = row + batch * mm.bias_stride;
            value += mm.bias_type == HIP_R_32F ? static_cast<const float*>(mm.bias)[index]
                                               : float(static_cast<const T*>(mm.bias)[index]);
        }
        if(mm.activation == HIPSPARSELT_MATMUL_ACTIVATION_RELU)
            value
                = value > mm.reluThreshold ? fminf(value, mm.reluUpper) : fminf(0.0f, mm.reluUpper);
        else if(mm.activation == HIPSPARSELT_MATMUL_ACTIVATION_ABS)
            value = fabsf(value);
        else if(mm.activation == HIPSPARSELT_MATMUL_ACTIVATION_LEAKYRELU && value < 0)
            value *= mm.leakyAlpha;
        else if(mm.activation == HIPSPARSELT_MATMUL_ACTIVATION_GELU)
            value = mm.geluScale * 0.5f * value
                    * (1.0f
                       + tanhf(0.7978845608028654f * value * (1.0f + 0.044715f * value * value)));
        else if(mm.activation == HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID)
            value = value >= 0 ? 1.0f / (1.0f + expf(-value)) : expf(value) / (1.0f + expf(value));
        else if(mm.activation == HIPSPARSELT_MATMUL_ACTIVATION_TANH)
            value = tanhf(value * mm.tanhAlpha) * mm.tanhBeta;
        if constexpr(std::is_same_v<T, int8_t>)
            // Upstream INT8 output uses nearest-even rounding then saturation.
            // Clamp before conversion also avoids undefined casts at infinity.
            value = nearbyintf(fminf(127.0f, fmaxf(-128.0f, value)));
        d[offset(mm.d, HIPSPARSE_OPERATION_NON_TRANSPOSE, row, col, batch)] = T(value);
    }

    template <bool Check, bool Compress>
    hipsparseStatus_t launch_strip(const hipsparseLtHandle_t* handle,
                                   Matrix                     matrix,
                                   int                        sparse_a,
                                   hipsparseOperation_t       op,
                                   const void*                input,
                                   void*                      output,
                                   int*                       invalid,
                                   hipStream_t                stream)
    {
        bool valid_handle = valid(handle);
        if(!valid_handle || !matrix.sparse || !operation(op) || (sparse_a != 0 && sparse_a != 1)
           || !input || (Check ? !invalid : !output))
        {
            if(std::getenv("HIPSPARSELT_SCALAR_DIAGNOSTICS"))
                std::fprintf(stderr,
                             "hipSPARSELt strip arguments: handle=%d structured=%d op=%d "
                             "sparse_a=%d input=%d output=%d invalid=%d rows=%lld cols=%lld\n",
                             int(valid_handle),
                             int(matrix.sparse),
                             int(op),
                             sparse_a,
                             int(input != nullptr),
                             int(output != nullptr),
                             int(invalid != nullptr),
                             static_cast<long long>(matrix.rows),
                             static_cast<long long>(matrix.cols));
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
        if(contraction(matrix, sparse_a, op) % group_size(matrix))
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        if constexpr(!Check)
        {
            size_t output_bytes = Compress ? compressed_bytes(matrix) : dense_bytes(matrix);
            if((Compress || input != output)
               && overlaps(input, dense_bytes(matrix), output, output_bytes))
                return HIPSPARSE_STATUS_INVALID_VALUE;
        }
        if(matrix.batch_stride == 0)
            matrix.batches = 1;
        if constexpr(Check)
        {
            if(hipMemsetAsync(invalid, 0, sizeof(int), stream) != hipSuccess)
                return HIPSPARSE_STATUS_EXECUTION_FAILED;
        }
        int64_t count = groups(matrix) * matrix.batches;
        dim3    grid((count + 255) / 256), block(256);
        if(matrix.type == HIP_R_32F)
            hipLaunchKernelGGL((strip_kernel<float, Check, Compress>),
                               grid,
                               block,
                               0,
                               stream,
                               matrix,
                               bool(sparse_a),
                               op,
                               (const float*)input,
                               (float*)output,
                               invalid);
        else if(matrix.type == HIP_R_16F)
            hipLaunchKernelGGL((strip_kernel<__half, Check, Compress>),
                               grid,
                               block,
                               0,
                               stream,
                               matrix,
                               bool(sparse_a),
                               op,
                               (const __half*)input,
                               (__half*)output,
                               invalid);
        else if(matrix.type == HIP_R_8I)
            hipLaunchKernelGGL((strip_kernel<int8_t, Check, Compress>),
                               grid,
                               block,
                               0,
                               stream,
                               matrix,
                               bool(sparse_a),
                               op,
                               (const int8_t*)input,
                               (int8_t*)output,
                               invalid);
        else
            hipLaunchKernelGGL((strip_kernel<hip_bfloat16, Check, Compress>),
                               grid,
                               block,
                               0,
                               stream,
                               matrix,
                               bool(sparse_a),
                               op,
                               (const hip_bfloat16*)input,
                               (hip_bfloat16*)output,
                               invalid);
        return hip_status(hipGetLastError());
    }
    hipsparseStatus_t init_matrix(const hipsparseLtHandle_t*  handle,
                                  hipsparseLtMatDescriptor_t* desc,
                                  int64_t                     r,
                                  int64_t                     c,
                                  int64_t                     ld,
                                  uint32_t                    alignment,
                                  hipDataType                 type,
                                  hipsparseOrder_t            order,
                                  bool                        structured)
    {
        if(!valid(handle) || !desc || r <= 0 || c <= 0 || ld <= 0 || !alignment
           || (alignment & (alignment - 1))
           || (order != HIPSPARSE_ORDER_ROW && order != HIPSPARSE_ORDER_COL))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        if(type != HIP_R_32F && type != HIP_R_16F && type != HIP_R_16BF && type != HIP_R_8I)
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        if(ld < (order == HIPSPARSE_ORDER_ROW ? c : r))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        // Bound all launch/index/size products before multiplication, including
        // user-selected batch strides; the fallback is intended for bounded work.
        constexpr int64_t limit = int64_t(1) << 30;
        if(r > limit || c > limit || ld > limit || r > limit / c
           || ld > limit / (order == HIPSPARSE_ORDER_ROW ? r : c))
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        Matrix matrix{};
        matrix.rows         = r;
        matrix.cols         = c;
        matrix.ld           = ld;
        matrix.batch_stride = ld * (order == HIPSPARSE_ORDER_ROW ? r : c);
        matrix.type         = type;
        matrix.order        = order;
        matrix.sparse       = structured;
        write(desc, matrix);
        return HIPSPARSE_STATUS_SUCCESS;
    }
}

extern "C" {
hipsparseStatus_t hipsparseLtInit(hipsparseLtHandle_t* opaque)
{
    if(!opaque)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    Handle          h;
    hipDeviceProp_t properties{};
    if(hipGetDevice(&h.device) != hipSuccess
       || hipGetDeviceProperties(&properties, h.device) != hipSuccess)
        return HIPSPARSE_STATUS_EXECUTION_FAILED;
    if(std::strncmp(properties.gcnArchName, "gfx1013", 7)
       || (properties.gcnArchName[7] && properties.gcnArchName[7] != ':'))
        return HIPSPARSE_STATUS_ARCH_MISMATCH;
    write(opaque, h);
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t* h)
{
    return destroy(h, handle_tag);
}
void              hipsparseLtInitialize() { }
hipsparseStatus_t hipsparseLtGetVersion(const hipsparseLtHandle_t* h, int* v)
{
    if(!valid(h) || !v)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    *v = hipsparseltVersionMajor * 100000 + hipsparseltVersionMinor * 100 + hipsparseltVersionPatch;
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtGetProperty(hipLibraryPropertyType p, int* v)
{
    if(!v)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    switch(p)
    {
    case HIP_LIBRARY_MAJOR_VERSION:
        *v = hipsparseltVersionMajor;
        break;
    case HIP_LIBRARY_MINOR_VERSION:
        *v = hipsparseltVersionMinor;
        break;
    case HIP_LIBRARY_PATCH_LEVEL:
        *v = hipsparseltVersionPatch;
        break;
    default:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtGetGitRevision(hipsparseLtHandle_t h, char* rev)
{
    if(!valid(&h) || !rev)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    std::strcpy(rev, HIPSPARSELT_STRINGIFY(hipsparseltVersionTweak));
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtGetArchName(char** name)
{
    if(!name)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    *name = nullptr;
    int             device;
    hipDeviceProp_t properties{};
    if(hipGetDevice(&device) != hipSuccess
       || hipGetDeviceProperties(&properties, device) != hipSuccess)
        return HIPSPARSE_STATUS_EXECUTION_FAILED;
    size_t length = std::strlen(properties.gcnArchName) + 1;
    *name         = static_cast<char*>(std::malloc(length));
    if(!*name)
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    std::memcpy(*name, properties.gcnArchName, length);
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtDenseDescriptorInit(const hipsparseLtHandle_t*  h,
                                                 hipsparseLtMatDescriptor_t* d,
                                                 int64_t                     r,
                                                 int64_t                     c,
                                                 int64_t                     ld,
                                                 uint32_t                    align,
                                                 hipDataType                 type,
                                                 hipsparseOrder_t            order)
{
    return init_matrix(h, d, r, c, ld, align, type, order, false);
}
hipsparseStatus_t hipsparseLtStructuredDescriptorInit(const hipsparseLtHandle_t*  h,
                                                      hipsparseLtMatDescriptor_t* d,
                                                      int64_t                     r,
                                                      int64_t                     c,
                                                      int64_t                     ld,
                                                      uint32_t                    align,
                                                      hipDataType                 type,
                                                      hipsparseOrder_t            order,
                                                      hipsparseLtSparsity_t       sparsity)
{
    if(sparsity != HIPSPARSELT_SPARSITY_50_PERCENT)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    return init_matrix(h, d, r, c, ld, align, type, order, true);
}
hipsparseStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t* d)
{
    return destroy(d, matrix_tag);
}
hipsparseStatus_t hipsparseLtMatDescSetAttribute(const hipsparseLtHandle_t*    h,
                                                 hipsparseLtMatDescriptor_t*   d,
                                                 hipsparseLtMatDescAttribute_t attr,
                                                 const void*                   value,
                                                 size_t                        bytes)
{
    Matrix matrix{};
    if(!valid(h) || !read(d, matrix, matrix_tag) || !value)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(attr == HIPSPARSELT_MAT_NUM_BATCHES && bytes == sizeof(int))
    {
        int n;
        std::memcpy(&n, value, bytes);
        if(n <= 0 || n > 65535 || matrix.rows * matrix.cols > (int64_t(1) << 30) / n)
            return HIPSPARSE_STATUS_INVALID_VALUE;
        matrix.batches = n;
    }
    else if(attr == HIPSPARSELT_MAT_BATCH_STRIDE && bytes == sizeof(int64_t))
    {
        int64_t stride;
        std::memcpy(&stride, value, bytes);
        int64_t span
            = matrix.ld * (matrix.order == HIPSPARSE_ORDER_ROW ? matrix.rows : matrix.cols);
        if((stride != 0 && stride < span) || stride > (int64_t(1) << 40) / 65535)
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        matrix.batch_stride = stride;
    }
    else
        return HIPSPARSE_STATUS_INVALID_VALUE;
    write(d, matrix);
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatDescGetAttribute(const hipsparseLtHandle_t*        h,
                                                 const hipsparseLtMatDescriptor_t* d,
                                                 hipsparseLtMatDescAttribute_t     attr,
                                                 void*                             value,
                                                 size_t                            bytes)
{
    Matrix m{};
    if(!valid(h) || !read(d, m, matrix_tag) || !value)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(attr == HIPSPARSELT_MAT_NUM_BATCHES && bytes == sizeof(int))
        std::memcpy(value, &m.batches, bytes);
    else if(attr == HIPSPARSELT_MAT_BATCH_STRIDE && bytes == sizeof(int64_t))
        std::memcpy(value, &m.batch_stride, bytes);
    else
        return HIPSPARSE_STATUS_INVALID_VALUE;
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmulDescriptorInit(const hipsparseLtHandle_t*        h,
                                                  hipsparseLtMatmulDescriptor_t*    d,
                                                  hipsparseOperation_t              opa,
                                                  hipsparseOperation_t              opb,
                                                  const hipsparseLtMatDescriptor_t* a,
                                                  const hipsparseLtMatDescriptor_t* b,
                                                  const hipsparseLtMatDescriptor_t* c,
                                                  const hipsparseLtMatDescriptor_t* out,
                                                  hipsparseLtComputetype_t          compute)
{
    Multiply mm{};
    if(!valid(h) || !d || !read(a, mm.a, matrix_tag) || !read(b, mm.b, matrix_tag)
       || !read(c, mm.c, matrix_tag) || !read(out, mm.d, matrix_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(!operation(opa) || !operation(opb)
       || compute != (mm.a.type == HIP_R_8I ? HIPSPARSELT_COMPUTE_32I : HIPSPARSELT_COMPUTE_32F))
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    mm.opa       = opa;
    mm.opb       = opb;
    mm.bias_type = mm.a.type == HIP_R_8I ? HIP_R_32F : mm.a.type;
    if(mm.a.sparse == mm.b.sparse || mm.c.sparse || mm.d.sparse || mm.a.type != mm.b.type
       || mm.a.type != mm.c.type || mm.a.type != mm.d.type)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    int64_t m = rows(mm.a, opa), n = cols(mm.b, opb), k = cols(mm.a, opa);
    if(k != rows(mm.b, opb) || mm.c.rows != m || mm.d.rows != m || mm.c.cols != n || mm.d.cols != n)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    // INT8 minimum product is -16256, maximum 16384. With 2:4 packing,
    // K<=262140 guarantees K/2*16384<=INT32_MAX, including partial sums.
    // Larger contractions and widened outputs remain deliberately unadmitted.
    if(mm.a.type == HIP_R_8I && k > 262140)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    if(k % group_size(sparse(mm)))
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    if(mm.a.batches != mm.d.batches || mm.b.batches != mm.d.batches || mm.c.batches != mm.d.batches)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    if(mm.d.batches > 1 && mm.d.batch_stride == 0)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    write(d, mm);
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmulDescSetAttribute(const hipsparseLtHandle_t*       h,
                                                    hipsparseLtMatmulDescriptor_t*   d,
                                                    hipsparseLtMatmulDescAttribute_t attr,
                                                    const void*                      value,
                                                    size_t                           bytes)
{
    Multiply mm{};
    if(!valid(h) || !read(d, mm, multiply_tag) || !value)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(attr == HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING)
    {
        if(bytes != sizeof(int))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        int enabled;
        std::memcpy(&enabled, value, bytes);
        if(enabled != 0 && enabled != 1)
            return HIPSPARSE_STATUS_INVALID_VALUE;
        return enabled ? HIPSPARSE_STATUS_NOT_SUPPORTED : HIPSPARSE_STATUS_SUCCESS;
    }
    if(attr == HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING)
    {
        if(bytes != sizeof(int))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(&mm.alpha_vector, value, bytes);
        if(mm.alpha_vector != 0 && mm.alpha_vector != 1)
            return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    else if(attr == HIPSPARSELT_MATMUL_BIAS_POINTER)
    {
        if(bytes != sizeof(mm.bias))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(&mm.bias, value, bytes);
    }
    else if(attr == HIPSPARSELT_MATMUL_BIAS_STRIDE)
    {
        if(bytes != sizeof(mm.bias_stride))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(&mm.bias_stride, value, bytes);
        if(mm.bias_stride != 0 && mm.bias_stride < mm.d.rows)
            return HIPSPARSE_STATUS_INVALID_VALUE;
        if(mm.bias_stride > (int64_t(1) << 40) / 65535)
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
    else if(attr == HIPSPARSELT_MATMUL_BIAS_TYPE)
    {
        if(bytes != sizeof(mm.bias_type))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(&mm.bias_type, value, bytes);
        if(mm.bias_type != HIP_R_32F && (mm.a.type == HIP_R_8I || mm.bias_type != mm.a.type))
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
    else if(activationAttribute(attr))
    {
        if(bytes != sizeof(int))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        int enabled;
        std::memcpy(&enabled, value, bytes);
        if(enabled != 0 && enabled != 1)
            return HIPSPARSE_STATUS_INVALID_VALUE;
        if(enabled && mm.d.type == HIP_R_8I
           && (attr == HIPSPARSELT_MATMUL_ACTIVATION_SIGMOID
               || attr == HIPSPARSELT_MATMUL_ACTIVATION_TANH))
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        mm.activation = enabled ? int(attr) : -1;
    }
    else if(auto parameter = activationParameter(mm, attr))
    {
        if(bytes != sizeof(float))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        float alpha;
        std::memcpy(&alpha, value, bytes);
        if(!std::isfinite(alpha)
           && !(attr == HIPSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND
                && alpha == std::numeric_limits<float>::infinity()))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        *parameter = alpha;
        if(attr == HIPSPARSELT_MATMUL_ACTIVATION_GELU_SCALING)
            mm.activation = HIPSPARSELT_MATMUL_ACTIVATION_GELU;
    }
    else
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    if(mm.bias && reinterpret_cast<uintptr_t>(mm.bias) % width(mm.bias_type))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    write(d, mm);
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmulDescGetAttribute(const hipsparseLtHandle_t*           h,
                                                    const hipsparseLtMatmulDescriptor_t* d,
                                                    hipsparseLtMatmulDescAttribute_t     attr,
                                                    void*                                value,
                                                    size_t                               bytes)
{
    Multiply mm{};
    if(!valid(h) || !read(d, mm, multiply_tag) || !value)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(attr == HIPSPARSELT_MATMUL_BETA_VECTOR_SCALING)
    {
        if(bytes != sizeof(int))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        int disabled = 0;
        std::memcpy(value, &disabled, bytes);
        return HIPSPARSE_STATUS_SUCCESS;
    }
    if(attr == HIPSPARSELT_MATMUL_ALPHA_VECTOR_SCALING)
    {
        if(bytes != sizeof(int))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(value, &mm.alpha_vector, bytes);
    }
    else if(attr == HIPSPARSELT_MATMUL_BIAS_POINTER)
    {
        if(bytes != sizeof(mm.bias))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(value, &mm.bias, bytes);
    }
    else if(attr == HIPSPARSELT_MATMUL_BIAS_STRIDE)
    {
        if(bytes != sizeof(mm.bias_stride))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(value, &mm.bias_stride, bytes);
    }
    else if(attr == HIPSPARSELT_MATMUL_BIAS_TYPE)
    {
        if(bytes != sizeof(mm.bias_type))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(value, &mm.bias_type, bytes);
    }
    else if(activationAttribute(attr))
    {
        if(bytes != sizeof(int))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        int enabled = mm.activation == int(attr);
        std::memcpy(value, &enabled, bytes);
    }
    else if(auto parameter = activationParameter(mm, attr))
    {
        if(bytes != sizeof(float))
            return HIPSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(value, parameter, bytes);
    }
    else
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t*           h,
                                                    hipsparseLtMatmulAlgSelection_t*     d,
                                                    const hipsparseLtMatmulDescriptor_t* desc,
                                                    hipsparseLtMatmulAlg_t               alg)
{
    Multiply mm{};
    if(!valid(h) || !d || !read(desc, mm, multiply_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(alg != HIPSPARSELT_MATMUL_ALG_DEFAULT)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    write(d, Algorithm{});
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmulAlgSelectionDestroy(const hipsparseLtMatmulAlgSelection_t* d)
{
    return destroy(d, algorithm_tag);
}
hipsparseStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t*       h,
                                                   hipsparseLtMatmulAlgSelection_t* d,
                                                   hipsparseLtMatmulAlgAttribute_t  attr,
                                                   const void*                      value,
                                                   size_t                           bytes)
{
    Algorithm a;
    if(!valid(h) || !read(d, a, algorithm_tag) || !value || bytes != sizeof(int))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    int v;
    std::memcpy(&v, value, bytes);
    if(attr == HIPSPARSELT_MATMUL_ALG_CONFIG_ID)
    {
        if(v != 0)
            return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    else if(attr == HIPSPARSELT_MATMUL_SEARCH_ITERATIONS)
    {
        if(v <= 0)
            return HIPSPARSE_STATUS_INVALID_VALUE;
        if(v != 1)
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
    }
    else
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    write(d, a);
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t*             h,
                                                   const hipsparseLtMatmulAlgSelection_t* d,
                                                   hipsparseLtMatmulAlgAttribute_t        attr,
                                                   void*                                  value,
                                                   size_t                                 bytes)
{
    Algorithm a;
    if(!valid(h) || !read(d, a, algorithm_tag) || !value || bytes != sizeof(int))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    int v;
    if(attr == HIPSPARSELT_MATMUL_ALG_CONFIG_ID)
        v = 0;
    else if(attr == HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID)
        v = 1;
    else if(attr == HIPSPARSELT_MATMUL_SEARCH_ITERATIONS)
        v = a.iterations;
    else
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    std::memcpy(value, &v, bytes);
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t*             h,
                                            hipsparseLtMatmulPlan_t*               out,
                                            const hipsparseLtMatmulDescriptor_t*   d,
                                            const hipsparseLtMatmulAlgSelection_t* alg)
{
    Plan      p;
    Algorithm a;
    if(!valid(h) || !out || !read(d, p.mm, multiply_tag) || !read(alg, a, algorithm_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    write(out, p);
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t* p)
{
    return destroy(p, plan_tag);
}
hipsparseStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t*     h,
                                                const hipsparseLtMatmulPlan_t* p,
                                                size_t*                        bytes)
{
    Plan plan;
    if(!valid(h) || !read(p, plan, plan_tag) || !bytes)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    *bytes = 0;
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t*     h,
                                    const hipsparseLtMatmulPlan_t* p,
                                    const void*                    alpha,
                                    const void*                    a,
                                    const void*                    b,
                                    const void*                    beta,
                                    const void*                    c,
                                    void*                          d,
                                    void*,
                                    hipStream_t* streams,
                                    int32_t      count)
{
    Plan plan;
    if(!valid(h) || !read(p, plan, plan_tag) || !alpha || !beta || !a || !b || !c || !d || count < 0
       || (count && !streams))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(count > 1)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    if(plan.mm.bias
       && (reinterpret_cast<uintptr_t>(plan.mm.bias) % width(plan.mm.bias_type)
           || overlaps(d, dense_bytes(plan.mm.d), plan.mm.bias, bias_bytes(plan.mm))))
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    if(overlaps(d,
                dense_bytes(plan.mm.d),
                a,
                plan.mm.a.sparse ? compressed_bytes(plan.mm.a) : dense_bytes(plan.mm.a))
       || overlaps(d,
                   dense_bytes(plan.mm.d),
                   b,
                   plan.mm.b.sparse ? compressed_bytes(plan.mm.b) : dense_bytes(plan.mm.b)))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(overlaps(d, dense_bytes(plan.mm.d), c, dense_bytes(plan.mm.c))
       && (d != c || plan.mm.c.order != plan.mm.d.order || plan.mm.c.ld != plan.mm.d.ld
           || plan.mm.c.batch_stride != plan.mm.d.batch_stride))
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    if(plan.mm.alpha_vector
       && (reinterpret_cast<uintptr_t>(alpha) % sizeof(float)
           || overlaps(d, dense_bytes(plan.mm.d), alpha, plan.mm.d.rows * sizeof(float))))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    float av = 1, bv;
    // Device vectors are read by the kernel; never dereference them on the host.
    if(!plan.mm.alpha_vector)
        std::memcpy(&av, alpha, sizeof(float));
    std::memcpy(&bv, beta, sizeof(float));
    hipStream_t stream = count ? streams[0] : nullptr;
    int64_t     total  = plan.mm.d.rows * plan.mm.d.cols * plan.mm.d.batches;
    dim3        grid((total + 255) / 256), block(256);
    if(plan.mm.a.type == HIP_R_32F)
        hipLaunchKernelGGL(multiply_kernel<float>,
                           grid,
                           block,
                           0,
                           stream,
                           plan.mm,
                           av,
                           bv,
                           plan.mm.alpha_vector ? static_cast<const float*>(alpha) : nullptr,
                           (const float*)a,
                           (const float*)b,
                           (const float*)c,
                           (float*)d);
    else if(plan.mm.a.type == HIP_R_16F)
        hipLaunchKernelGGL(multiply_kernel<__half>,
                           grid,
                           block,
                           0,
                           stream,
                           plan.mm,
                           av,
                           bv,
                           plan.mm.alpha_vector ? static_cast<const float*>(alpha) : nullptr,
                           (const __half*)a,
                           (const __half*)b,
                           (const __half*)c,
                           (__half*)d);
    else if(plan.mm.a.type == HIP_R_8I)
        hipLaunchKernelGGL(multiply_kernel<int8_t>,
                           grid,
                           block,
                           0,
                           stream,
                           plan.mm,
                           av,
                           bv,
                           plan.mm.alpha_vector ? static_cast<const float*>(alpha) : nullptr,
                           (const int8_t*)a,
                           (const int8_t*)b,
                           (const int8_t*)c,
                           (int8_t*)d);
    else
        hipLaunchKernelGGL(multiply_kernel<hip_bfloat16>,
                           grid,
                           block,
                           0,
                           stream,
                           plan.mm,
                           av,
                           bv,
                           plan.mm.alpha_vector ? static_cast<const float*>(alpha) : nullptr,
                           (const hip_bfloat16*)a,
                           (const hip_bfloat16*)b,
                           (const hip_bfloat16*)c,
                           (hip_bfloat16*)d);
    return hip_status(hipGetLastError());
}
hipsparseStatus_t hipsparseLtMatmulSearch(const hipsparseLtHandle_t* h,
                                          hipsparseLtMatmulPlan_t*   p,
                                          const void*                alpha,
                                          const void*                a,
                                          const void*                b,
                                          const void*                beta,
                                          const void*                c,
                                          void*                      d,
                                          void*                      workspace,
                                          hipStream_t*               streams,
                                          int32_t                    count)
{
    auto status = hipsparseLtMatmul(h, p, alpha, a, b, beta, c, d, workspace, streams, count);
    if(status != HIPSPARSE_STATUS_SUCCESS)
        return status;
    // The public Search API is blocking on streams[0], unlike Matmul.
    return hip_status(hipStreamSynchronize(count ? streams[0] : nullptr));
}
hipsparseStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t*        h,
                                         const hipsparseLtMatDescriptor_t* d,
                                         int                               sa,
                                         hipsparseOperation_t              op,
                                         const void*                       in,
                                         void*                             out,
                                         hipsparseLtPruneAlg_t             alg,
                                         hipStream_t                       stream)
{
    Matrix m{};
    if(!read(d, m, matrix_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(alg == HIPSPARSELT_PRUNE_SPMMA_TILE)
        return launch_tile(h, m, sa, op, in, out, stream);
    if(alg != HIPSPARSELT_PRUNE_SPMMA_STRIP)
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    return launch_strip<false, false>(h, m, sa, op, in, out, nullptr, stream);
}
hipsparseStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t*        h,
                                              const hipsparseLtMatDescriptor_t* d,
                                              int                               sa,
                                              hipsparseOperation_t              op,
                                              const void*                       in,
                                              int*                              invalid,
                                              hipStream_t                       stream)
{
    Matrix m{};
    if(!read(d, m, matrix_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    return launch_strip<true, false>(h, m, sa, op, in, nullptr, invalid, stream);
}
hipsparseStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t*           h,
                                        const hipsparseLtMatmulDescriptor_t* d,
                                        const void*                          in,
                                        void*                                out,
                                        hipsparseLtPruneAlg_t                alg,
                                        hipStream_t                          stream)
{
    Multiply mm{};
    if(!read(d, mm, multiply_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    hipsparseLtMatDescriptor_t desc{};
    write(&desc, sparse(mm));
    return hipsparseLtSpMMAPrune2(h, &desc, mm.a.sparse, sparse_op(mm), in, out, alg, stream);
}
hipsparseStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t*           h,
                                             const hipsparseLtMatmulDescriptor_t* d,
                                             const void*                          in,
                                             int*                                 invalid,
                                             hipStream_t                          stream)
{
    Multiply mm{};
    if(!read(d, mm, multiply_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    hipsparseLtMatDescriptor_t desc{};
    write(&desc, sparse(mm));
    return hipsparseLtSpMMAPruneCheck2(h, &desc, mm.a.sparse, sparse_op(mm), in, invalid, stream);
}
hipsparseStatus_t hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t*        h,
                                                  const hipsparseLtMatDescriptor_t* d,
                                                  size_t*                           bytes,
                                                  size_t*                           workspace)
{
    Matrix m{};
    if(!valid(h) || !read(d, m, matrix_tag) || !m.sparse || !bytes || !workspace)
        return HIPSPARSE_STATUS_INVALID_VALUE;
    if(m.rows * m.cols % group_size(m))
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    *bytes     = compressed_bytes(m);
    *workspace = 0;
    return HIPSPARSE_STATUS_SUCCESS;
}
hipsparseStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t*        h,
                                            const hipsparseLtMatDescriptor_t* d,
                                            int                               sa,
                                            hipsparseOperation_t              op,
                                            const void*                       in,
                                            void*                             out,
                                            void*,
                                            hipStream_t stream)
{
    Matrix m{};
    if(!read(d, m, matrix_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    return launch_strip<false, true>(h, m, sa, op, in, out, nullptr, stream);
}
hipsparseStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t*     h,
                                                 const hipsparseLtMatmulPlan_t* p,
                                                 size_t*                        bytes,
                                                 size_t*                        workspace)
{
    Plan plan;
    if(!read(p, plan, plan_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    hipsparseLtMatDescriptor_t desc{};
    write(&desc, sparse(plan.mm));
    return hipsparseLtSpMMACompressedSize2(h, &desc, bytes, workspace);
}
hipsparseStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t*     h,
                                           const hipsparseLtMatmulPlan_t* p,
                                           const void*                    in,
                                           void*                          out,
                                           void*                          workspace,
                                           hipStream_t                    stream)
{
    Plan plan;
    if(!read(p, plan, plan_tag))
        return HIPSPARSE_STATUS_INVALID_VALUE;
    hipsparseLtMatDescriptor_t desc{};
    write(&desc, sparse(plan.mm));
    return hipsparseLtSpMMACompress2(
        h, &desc, plan.mm.a.sparse, sparse_op(plan.mm), in, out, workspace, stream);
}
}
