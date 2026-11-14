#include <torch/extension.h>
#include <cuda_runtime.h>

// v2: 每线程算8x8, 结果攒在寄存器
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void gemm_regtile(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;  // 0~255
    int row0 = blockIdx.y * BM + ty * TM;
    int col0 = blockIdx.x * BN + tx * TN;

    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    for (int kb = 0; kb < K; kb += BK) {
        // 256线程分摊1024个元素的载入
        for (int i = 0; i < BM * BK / 256; i++) {
            int idx = tid + i * 256;
            int ar = idx / BK, ac = idx % BK;
            int gr = blockIdx.y * BM + ar;
            As[ar][ac] = (gr < M && kb + ac < K) ? A[gr * K + kb + ac] : 0.0f;
        }
        for (int i = 0; i < BK * BN / 256; i++) {
            int idx = tid + i * 256;
            int br = idx / BN, bc = idx % BN;
            int gc = blockIdx.x * BN + bc;
            Bs[br][bc] = (kb + br < K && gc < N) ? B[(kb + br) * N + gc] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            float ra[TM], rb[TN];
            for (int i = 0; i < TM; i++) ra[i] = As[ty * TM + i][k];
            for (int j = 0; j < TN; j++) rb[j] = Bs[k][tx * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++) acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int row = row0 + i;
        if (row >= M) break;
        for (int j = 0; j < TN; j++) {
            int col = col0 + j;
            if (col < N) C[row * N + col] = acc[i][j];
        }
    }
}

void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_regtile<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm);
}
