#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

// v3: cp.async双缓冲, 边拷下一块边算当前块
// 要求 M%128==0, N%128==0, K%8==0, 由python侧pad保证
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__device__ void load_tile_async(const float* A, const float* B,
                                int M, int N, int K, int kb,
                                int bx, int by, int buf, int tid,
                                float (*As)[BM][BK], float (*Bs)[BK][BN]) {
    // A tile 128x8, 每线程拷16B
    if (tid < 256) {
        int row = tid >> 1, half = tid & 1;
        const float* src = A + (by * BM + row) * K + kb + half * 4;
        __pipeline_memcpy_async(&As[buf][row][half * 4], src, 16);
    }
    // B tile 8x128
    if (tid < 256) {
        int row = tid >> 5, chunk = tid & 31;
        const float* src = B + (kb + row) * N + bx * BN + chunk * 4;
        __pipeline_memcpy_async(&Bs[buf][row][chunk * 4], src, 16);
    }
    __pipeline_commit();
}

__global__ void gemm_double_buf(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int row0 = blockIdx.y * BM + ty * TM;
    int col0 = blockIdx.x * BN + tx * TN;

    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;

    load_tile_async(A, B, M, N, K, 0, blockIdx.x, blockIdx.y, 0, tid, As, Bs);

    int buf = 0;
    for (int kb = 0; kb < K; kb += BK, buf ^= 1) {
        if (kb + BK < K)  // 预取下一块
            load_tile_async(A, B, M, N, K, kb + BK, blockIdx.x, blockIdx.y,
                            buf ^ 1, tid, As, Bs);
        else
            __pipeline_commit();
        __pipeline_wait_prior(1);  // 当前块已就位
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            float ra[TM], rb[TN];
            for (int i = 0; i < TM; i++) ra[i] = As[buf][ty * TM + i][k];
            for (int j = 0; j < TN; j++) rb[j] = Bs[buf][k][tx * TN + j];
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
    dim3 grid(N / BN, M / BM);
    gemm_double_buf<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm);
}
