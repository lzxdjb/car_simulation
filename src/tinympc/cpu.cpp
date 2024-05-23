
using namespace std;

#define NSTATES 12
#define NINPUTS 4

#define NHORIZON 10
#define NTOTAL 301

#include <torch/extension.h>
#include<Eigen.h>
#include "problem_data/quadrotor_20hz_params.hpp"
#include <tinympc/admm.hpp>

// template <typename scalar_t, uint32_t N_DIMS>

void graph_max_cpu(float* data) 
{
    cout<<"asdfasdf";

    for (int i = 0; i < 3; ++i) {
        data[i] = i + 1.0;  // Appending [1.1, 2.0, 3.0]
    }
  
}


at::Tensor cpu(torch::Tensor data_s) {

    torch::Tensor append_tensor = torch::tensor({1.1 , 2.0 , 3.0});  
    data_s = torch::cat({data_s, append_tensor}, 0);

    auto point_index = data_s.new_zeros({128});
    cout<<point_index<<endl;

    graph_max_cpu(point_index.data<float>());
    cout<<"after that"<<point_index<<endl;

    return point_index;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu", &cpu, "Graph Max CUDA");
}
