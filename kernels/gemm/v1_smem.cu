#include <torch/extension.h>
#include <cuda_runtime.h>

// v1: 分块加载到共享内存
#define BM 32
#define BN 32
#define BK 32

__global__ void gemm_smem(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;
    float acc = 0.0f;

    for (int kb = 0; kb < K; kb += BK) {
        if (row < M && kb + tx < K) As[ty][tx] = A[row * K + kb + tx];
        else As[ty][tx] = 0.0f;  // 越界补零
        if (kb + ty < K && col < N) Bs[ty][tx] = B[(kb + ty) * N + col];
        else Bs[ty][tx] = 0.0f;
        __syncthreads();  // 等tile就位
        for (int k = 0; k < BK; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // 防止提前覆盖
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_smem<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm);
}
