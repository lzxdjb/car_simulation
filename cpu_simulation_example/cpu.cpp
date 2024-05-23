
using namespace std;

#define NSTATES 12
#define NINPUTS 4

#define NHORIZON 10
#define NTOTAL 301

#include <torch/extension.h>
#include <Eigen.h>
#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"
#include <tinympc/admm.hpp>

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

typedef Matrix<tinytype, NINPUTS, NHORIZON - 1> tiny_MatrixNuNhm1;
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

void graph_max_cpu(double *data)
{
    cache.rho = rho_value;
    cache.Kinf = Map<Matrix<tinytype, NINPUTS, NSTATES, RowMajor>>(Kinf_data);
    cache.Pinf = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Pinf_data);
    cache.Quu_inv = Map<Matrix<tinytype, NINPUTS, NINPUTS, RowMajor>>(Quu_inv_data);
    cache.AmBKt = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(AmBKt_data);

    work.Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    work.Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    work.Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    work.R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    work.nx = NSTATES;
    work.nu = NINPUTS;
    work.N = NHORIZON;

    work.u_min = tiny_MatrixNuNhm1::Constant(-0.5);
    work.u_max = tiny_MatrixNuNhm1::Constant(0.5);
    work.x_min = tiny_MatrixNxNh::Constant(-5);
    work.x_max = tiny_MatrixNxNh::Constant(5);

    work.Xref = tiny_MatrixNxNh::Zero();
    work.Uref = tiny_MatrixNuNhm1::Zero();

    work.x = tiny_MatrixNxNh::Zero();
    work.q = tiny_MatrixNxNh::Zero();
    work.p = tiny_MatrixNxNh::Zero();
    work.v = tiny_MatrixNxNh::Zero();
    work.vnew = tiny_MatrixNxNh::Zero();
    work.g = tiny_MatrixNxNh::Zero();

    work.u = tiny_MatrixNuNhm1::Zero();
    work.r = tiny_MatrixNuNhm1::Zero();
    work.d = tiny_MatrixNuNhm1::Zero();
    work.z = tiny_MatrixNuNhm1::Zero();
    work.znew = tiny_MatrixNuNhm1::Zero();
    work.y = tiny_MatrixNuNhm1::Zero();

    work.primal_residual_state = 0;
    work.primal_residual_input = 0;
    work.dual_residual_state = 0;
    work.dual_residual_input = 0;
    work.status = 0;
    work.iter = 0;

    settings.abs_pri_tol = 0.001;
    settings.abs_dua_tol = 0.001;
    settings.max_iter = 100;
    settings.check_termination = 1;
    settings.en_input_bound = 1;
    settings.en_state_bound = 1;

    tiny_VectorNx x0, x1; // current and next simulation states

    // Map data from trajectory_data
    Matrix<tinytype, NSTATES, NTOTAL> Xref_total = Eigen::Map<Matrix<tinytype, NSTATES, NTOTAL>>(Xref_data);
    work.Xref = Xref_total.block<NSTATES, NHORIZON>(0, 0);

    // Initial state
    x0 = work.Xref.col(0);
    // cout<<"asdfasdf";

    // for (int i = 0; i < 3; ++i)
    // {
    //     data[i] = i + 1.0; // Appending [1.1, 2.0, 3.0]
    // }

    for (int k = 0; k < NTOTAL - NHORIZON; ++k)
    {
        std::cout << "tracking error: " << (x0 - work.Xref.col(1)).norm() << std::endl;

        // 1. Update measurement
        work.x.col(0) = x0;

        // 2. Update reference
        work.Xref = Xref_total.block<NSTATES, NHORIZON>(0, k);

        // 3. Reset dual variables if needed
        work.y = tiny_MatrixNuNhm1::Zero();
        work.g = tiny_MatrixNxNh::Zero();

        // 4. Solve MPC problem
        tiny_solve(&solver);

        // std::cout << work.iter << std::endl;
        // std::cout << work.u.col(0).transpose().format(CleanFmt) << std::endl;

        // 5. Simulate forward
        x1 = work.Adyn * x0 + work.Bdyn * work.u.col(0);
        x0 = x1;

        // std::cout << x0.transpose().format(CleanFmt) << std::endl;
        // std::cout<<x0<<endl;

      
        int start_index = k * tiny_VectorNx::RowsAtCompileTime;
        for (int j = 0; j < tiny_VectorNx::RowsAtCompileTime; ++j) {
            data[start_index + j] = x0[j];
        }
    
    }


    // quick test

    // for (int i = 0; i < x0.size(); ++i) {
    //     x0[i] = static_cast<tinytype>(i);
    // }

    // for (int i = 0; i < 1; ++i) {
    //     int start_index = i * tiny_VectorNx::RowsAtCompileTime;
    //     for (int j = 0; j < tiny_VectorNx::RowsAtCompileTime; ++j) {
    //         data[start_index + j] = x0[j];
    //     }
    // }
}

at::Tensor cpu(torch::Tensor data_s)
{

    torch::Tensor append_tensor = torch::tensor({1.1, 2.0, 3.0});
    data_s = torch::cat({data_s, append_tensor}, 0);

    auto point_index = data_s.new_zeros({400 , 12});
    // cout<<point_index<<endl;

    graph_max_cpu(point_index.data_ptr<double>());
    // cout << "after that" << point_index << endl;

    return point_index;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cpu", &cpu, "Graph Max CUDA");
}
