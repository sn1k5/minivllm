#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

// v4: wmma tensor core, fp16进fp32累加
// 要求 M%64==0, N%64==0, K%16==0, 由python侧pad保证
using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 16
#define LDS_A 24  // 16加8防bank冲突
#define LDS_B 72  // 64加8

__global__ void gemm_wmma(const __half* A, const __half* B, __half* C,
                          int M, int N, int K) {
    __shared__ __half As[2][BM][LDS_A];
    __shared__ __half Bs[2][BK][LDS_B];
    int tid = threadIdx.x;
    int warp = tid / 32;
    int wy = warp / 4, wx = warp % 4;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // 每线程拷一段16B, A共128段B共128段
    auto load = [&](int kb, int buf) {
        if (tid < 128) {
            int row = tid >> 1, half = tid & 1;
            const __half* src = A + (blockIdx.y * BM + row) * K + kb + half * 8;
            __pipeline_memcpy_async(&As[buf][row][half * 8], src, 16);
        } else if (tid < 256) {
            int c = tid - 128;
            int row = c >> 3, col = (c & 7) * 8;
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

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, &As[buf][wy * 16][0], LDS_A);
        wmma::load_matrix_sync(fb, &Bs[buf][0][wx * 16], LDS_B);
        wmma::mma_sync(acc, fa, fb, acc);
        __syncthreads();
    }

    // fp32结果转回fp16再写
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc16;
    for (int i = 0; i < acc.num_elements; i++) {
        acc16.x[i] = __float2half(acc.x[i]);
    }
    int row = blockIdx.y * BM + wy * 16;
    int col = blockIdx.x * BN + wx * 16;
    wmma::store_matrix_sync(&C[row * N + col], acc16, N, wmma::mem_row_major);
}

void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(512);
    dim3 grid(N / BN, M / BM);
    gemm_wmma<<<grid, block>>>(
        (const __half*)A.data_ptr(), (const __half*)B.data_ptr(),
        (__half*)C.data_ptr(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm);
}
