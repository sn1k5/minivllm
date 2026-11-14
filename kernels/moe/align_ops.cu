#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// vllm moe_align_block_size的CUDA实现, 语义对齐, 供对比测试用
__global__ void count_kernel(const int* topk_ids, int* counts, int total) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < total) atomicAdd(&counts[topk_ids[p]], 1);
}

// 单线程算pad偏移/每block专家号/总pad数
__global__ void plan_kernel(const int* counts, int* poff, int* expert_ids,
                            int* total_pad, int E, int bs) {
    if (threadIdx.x != 0) return;
    int acc = 0, nb = 0;
    for (int e = 0; e < E; e++) {
        poff[e] = acc;
        int nb_e = (counts[e] + bs - 1) / bs;
        for (int b = 0; b < nb_e; b++) expert_ids[nb++] = e;
        acc += nb_e * bs;
    }
    *total_pad = acc;
}

__global__ void scatter_kernel(const int* topk_ids, const int* poff,
                               int* cursor, int* sorted, int total) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total) return;
    int e = topk_ids[p];
    int pos = atomicAdd(&cursor[e], 1);
    sorted[poff[e] + pos] = p;
}

std::vector<torch::Tensor> moe_align(torch::Tensor topk_ids, int64_t E,
                                     int64_t block_size) {
    int total = topk_ids.numel();
    int bs = block_size;
    int max_pad = total + E * (bs - 1);
    int max_mblk = (max_pad + bs - 1) / bs;
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
    auto sorted_ids = torch::full({max_pad}, total, opts);
    auto expert_ids = torch::zeros({max_mblk}, opts);
    auto num_post = torch::zeros({1}, opts);
    auto counts = torch::zeros({E}, opts);
    auto cursor = torch::zeros({E}, opts);
    auto poff = torch::empty({E}, opts);
    count_kernel<<<(total + 255) / 256, 256>>>(
        topk_ids.data_ptr<int>(), counts.data_ptr<int>(), total);
    plan_kernel<<<1, 32>>>(counts.data_ptr<int>(), poff.data_ptr<int>(),
                           expert_ids.data_ptr<int>(), num_post.data_ptr<int>(),
                           E, bs);
    scatter_kernel<<<(total + 255) / 256, 256>>>(
        topk_ids.data_ptr<int>(), poff.data_ptr<int>(), cursor.data_ptr<int>(),
        sorted_ids.data_ptr<int>(), total);
    return {sorted_ids, expert_ids, num_post};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_align", &moe_align);
}
