#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

// v4b: BK=32, 减少K循环次数, 大M更优
// 要求 M%64==0, N%64==0, K%32==0
using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 32
#define LDS_A 40  // 32加8防bank
#define LDS_B 72  // 64加8

__global__ void gemm_wmma_bk32(const __half* A, const __half* B, __half* C,
                               int M, int N, int K) {
    __shared__ __half As[2][BM][LDS_A];
    __shared__ __half Bs[2][BK][LDS_B];
    int tid = threadIdx.x;
    int warp = tid / 32;
    int wy = warp / 4, wx = warp % 4;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // A 64x32: 256线程每线程拷8个half(16B)
    auto load = [&](int kb, int buf) {
        if (tid < 256) {
            int row = tid >> 2, col = (tid & 3) * 8;
            const __half* src = A + (blockIdx.y * BM + row) * K + kb + col;
            __pipeline_memcpy_async(&As[buf][row][col], src, 16);
        }
        if (tid < 256) {
            int row = tid >> 3, col = (tid & 7) * 8;
            const __half* src = B + (kb + row) * N + blockIdx.x * BN + col;
            __pipeline_memcpy_async(&Bs[buf][row][col], src, 16);
        }
        __pipeline_commit();
    };

    load(0, 0);
    int buf = 0;
    for (int kb = 0; kb < K; kb += BK, buf ^= 1) {
        if (kb + BK < K) load(kb + BK, buf ^ 1);
        else __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();

        // 两个16宽K-tile
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa0, fa1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb0, fb1;
        wmma::load_matrix_sync(fa0, &As[buf][wy * 16][0], LDS_A);
        wmma::load_matrix_sync(fa1, &As[buf][wy * 16][16], LDS_A);
        wmma::load_matrix_sync(fb0, &Bs[buf][0][wx * 16], LDS_B);
        wmma::load_matrix_sync(fb1, &Bs[buf][16][wx * 16], LDS_B);
        wmma::mma_sync(acc, fa0, fb0, acc);
        wmma::mma_sync(acc, fa1, fb1, acc);
        __syncthreads();
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc16;
    for (int i = 0; i < acc.num_elements; i++)
        acc16.x[i] = __float2half(acc.x[i]);
    int row = blockIdx.y * BM + wy * 16;
    int col = blockIdx.x * BN + wx * 16;
    wmma::store_matrix_sync(&C[row * N + col], acc16, N, wmma::mem_row_major);
}

void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(512);
    dim3 grid(N / BN, M / BM);
    gemm_wmma_bk32<<<grid, block>>>(
        (const __half*)A.data_ptr(), (const __half*)B.data_ptr(),
        (__half*)C.data_ptr(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm);
}
