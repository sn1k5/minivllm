#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v2: BM=32分组GEMM + half2向量化, 减少padding浪费
// 对齐Qwen3-MoE: softmax->topk->renorm->专家FFN->加权和
#define BM 32
#define BN 64
#define BK 32

// 每个token一个block, E固定不超过128
__global__ void route_topk(const float* gating, int* topk_ids, float* topk_w,
                           int E, int K) {
    int t = blockIdx.x;
    int e = threadIdx.x;
    __shared__ float p[128];
    if (e < E) p[e] = gating[t * E + e];
    __syncthreads();
    // 每线程暴力求max, E小够用
    float mx = p[0];
    for (int i = 1; i < E; i++) mx = fmaxf(mx, p[i]);
    float v = __expf(p[e] - mx);
    __syncthreads();  // 等所有线程读完p再写
    p[e] = v;  // 只存exp, renorm在选择后做
    __syncthreads();
    if (e == 0) {
        float total = 0.0f;
        for (int k = 0; k < K; k++) {
            int best = 0;
            for (int i = 1; i < E; i++)
                if (p[i] > p[best]) best = i;
            topk_ids[t * K + k] = best;
            topk_w[t * K + k] = p[best];
            total += p[best];
            p[best] = -1.0f;  // 排除已选
        }
        for (int k = 0; k < K; k++) topk_w[t * K + k] /= total;
    }
}

__global__ void count_expert(const int* topk_ids, int* counts, int total) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < total) atomicAdd(&counts[topk_ids[p]], 1);
}

// 单线程算pad后的偏移和每block的专家号
__global__ void plan_offsets(const int* counts, int* poff, int* expert_ids,
                             int* num_blocks, int E, int bm) {
    if (threadIdx.x != 0) return;
    int acc = 0, nb = 0;
    for (int e = 0; e < E; e++) {
        poff[e] = acc;
        int cnt = counts[e];
        int pad = (cnt + bm - 1) / bm * bm;
        for (int b = 0; b < (cnt + bm - 1) / bm; b++) expert_ids[nb++] = e;
        acc += pad;
    }
    poff[E] = acc;
    *num_blocks = nb;
}

__global__ void scatter_tokens(const int* topk_ids, const int* poff,
                               int* cursor, int* sorted, int total) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total) return;
    int e = topk_ids[p];
    int pos = atomicAdd(&cursor[e], 1);
    sorted[poff[e] + pos] = p;
}

// 分组GEMM, A行按sorted间接寻址, is_w2时加权累加到out
// w13的A行取token, w2的A行取槽位本身(a_by_slot=1)
__global__ void moe_gemm(const __half* X, const __half* W,
                         const int* sorted, const int* expert_ids,
                         const int* poff, const int* num_blocks,
                         const float* topk_w, float* out_f,
                         __half* inter, int N, int K2, int H, int topk,
                         int a_by_slot) {
    int by = blockIdx.y;
    if (by >= *num_blocks) return;
    int e = expert_ids[by];
    int row0 = by * BM;
    int n0 = blockIdx.x * BN;
    int tid = threadIdx.x;
    int tx = tid % 32, ty = tid / 32;

    __shared__ __half As[BK][BM];
    __shared__ __half Bs[BN][BK];
    float acc[8][2];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 2; j++) acc[i][j] = 0.0f;

    for (int kb = 0; kb < K2; kb += BK) {
        // 加载A: BK*BM=1024个half, 128线程每线程8个
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * 128;
            int k = idx / BM, r = idx % BM;
            int pair = sorted[row0 + r];
            __half v = __float2half(0.0f);
            if (pair >= 0) {
                int arow = a_by_slot ? (row0 + r) : (pair / topk);
                v = X[arow * K2 + kb + k];
            }
            As[k][r] = v;
        }
        // 加载B: BN*BK=2048个half, 128线程每线程16个
        for (int i = 0; i < 16; i++) {
            int idx = tid + i * 128;
            int n = idx / BK, k = idx % BK;
            __half v = __float2half(0.0f);
            if (n0 + n < N && kb + k < K2)
                v = W[(long long)e * N * K2 + (n0 + n) * K2 + kb + k];
            Bs[n][k] = v;
        }
        __syncthreads();
        // half2向量化乘加
        for (int k = 0; k < BK; k++) {
            __half ra[8], rb[2];
            for (int i = 0; i < 8; i++) ra[i] = As[k][ty * 8 + i];
            for (int j = 0; j < 2; j++) rb[j] = Bs[tx * 2 + j][k];
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 2; j++)
                    acc[i][j] += __half2float(ra[i]) * __half2float(rb[j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 8; i++) {
        int slot = row0 + ty * 8 + i;
        int pair = sorted[slot];
        if (pair < 0) continue;  // pad行跳过
        for (int j = 0; j < 2; j++) {
            int col = n0 + tx * 2 + j;
            if (col >= N) continue;
            if (out_f != nullptr) {
                float wgt = topk_w[pair];
                atomicAdd(&out_f[(pair / topk) * H + col],
                          acc[i][j] * wgt);
            } else {
                inter[slot * N + col] = __float2half(acc[i][j]);
            }
        }
    }
}

// silu(gate)*up, 行数从poff尾部读
__global__ void silu_mul(const __half* inter, __half* act,
                         const int* total_rows, int I) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (*total_rows) * I;
    if (idx >= total) return;
    int r = idx / I, c = idx % I;
    float g = __half2float(inter[r * 2 * I + c]);
    float u = __half2float(inter[r * 2 * I + I + c]);
    float s = g / (1.0f + __expf(-g));
    act[idx] = __float2half(s * u);
}

__global__ void cast_out(const float* out_f, __half* out, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = __float2half(out_f[i]);
}

void moe_forward(torch::Tensor hidden, torch::Tensor w13, torch::Tensor w2,
                 torch::Tensor gating, int64_t topk, torch::Tensor out) {
    int T = hidden.size(0), H = hidden.size(1);
    int E = gating.size(1);
    int N1 = w13.size(1), I = w2.size(2);  // w2是[E,H,I]
    int K = topk;
    int total = T * K;
    int max_padded = total + E * BM;
    int max_blocks = (total + BM - 1) / BM + E;

    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(hidden.device());
    auto ids = torch::empty({T, K}, opts_i);
    auto counts = torch::zeros({E}, opts_i);
    auto cursor = torch::zeros({E}, opts_i);
    auto poff = torch::empty({E + 1}, opts_i);
    auto expert_ids = torch::empty({max_blocks}, opts_i);
    auto num_blocks = torch::empty({1}, opts_i);
    auto sorted = torch::full({max_padded}, -1, opts_i);
    auto topk_w = torch::empty({T, K}, torch::dtype(torch::kFloat32).device(hidden.device()));
    auto inter = torch::empty({max_padded, N1}, hidden.options());
    auto act = torch::empty({max_padded, I}, hidden.options());
    auto out_f = torch::zeros({T, H}, torch::dtype(torch::kFloat32).device(hidden.device()));

    route_topk<<<T, 128>>>(gating.data_ptr<float>(), ids.data_ptr<int>(),
                           topk_w.data_ptr<float>(), E, K);
    count_expert<<<(total + 255) / 256, 256>>>(ids.data_ptr<int>(),
                                               counts.data_ptr<int>(), total);
    plan_offsets<<<1, 32>>>(counts.data_ptr<int>(), poff.data_ptr<int>(),
                            expert_ids.data_ptr<int>(), num_blocks.data_ptr<int>(),
                            E, BM);
    scatter_tokens<<<(total + 255) / 256, 256>>>(
        ids.data_ptr<int>(), poff.data_ptr<int>(), cursor.data_ptr<int>(),
        sorted.data_ptr<int>(), total);

    dim3 grid1(N1 / BN, max_blocks);
    moe_gemm<<<grid1, 128>>>(
        (const __half*)hidden.data_ptr(), (const __half*)w13.data_ptr(),
        sorted.data_ptr<int>(), expert_ids.data_ptr<int>(), poff.data_ptr<int>(),
        num_blocks.data_ptr<int>(), nullptr, nullptr,
        (__half*)inter.data_ptr(), N1, H, H, K, 0);
    int rows = max_padded;
    silu_mul<<<(rows * I + 255) / 256, 256>>>(
        (__half*)inter.data_ptr(), (__half*)act.data_ptr(),
        poff.data_ptr<int>() + E, I);
    dim3 grid2(H / BN, max_blocks);
    moe_gemm<<<grid2, 128>>>(
        (const __half*)act.data_ptr(), (const __half*)w2.data_ptr(),
        sorted.data_ptr<int>(), expert_ids.data_ptr<int>(), poff.data_ptr<int>(),
        num_blocks.data_ptr<int>(), topk_w.data_ptr<float>(),
        out_f.data_ptr<float>(), nullptr, H, I, H, K, 1);
    cast_out<<<(T * H + 255) / 256, 256>>>(
        out_f.data_ptr<float>(), (__half*)out.data_ptr(), T * H);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_forward", &moe_forward);
}
