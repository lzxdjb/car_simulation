#include <torch/extension.h>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cmath>
// sdsds
void graph_max_cuda(const torch::Tensor data, torch::Tensor max_val);

at::Tensor graph_max(torch::Tensor data) {
    
    std::cout<<"asdfasdf";
    return data;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("graph_max", &graph_max, "Graph Max CUDA");
}
