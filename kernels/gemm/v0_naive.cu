#include <torch/extension.h>
#include <cuda_runtime.h>

// v0: 一个线程算一个C元素
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];  // B列访问不连续
    }
    C[row * N + col] = acc;
}

void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    gemm_naive<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm);
}
