#include <torch/extension.h>
using namespace at;
// sdsd
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void graph_max_kernel(const float* data, float* max_val, int64_t num_nodes) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_nodes) {
        shared_data[tid] = data[i];
    } else {
        shared_data[tid] = -FLT_MAX;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(max_val, shared_data[0]);
    }
}

at::Tensor graph_max_cuda(torch::Tensor data, torch ::Tensor max_val) {
    const int threads = 1024;
    const int blocks = (data.size(0) + threads - 1) / threads;
    const int shared_memory = threads * sizeof(float);

    graph_max_kernel<<<blocks, threads, shared_memory>>>(data.data_ptr<float>(), max_val.data_ptr<float>(), data.size(0));
}
