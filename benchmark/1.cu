#include "./taskflow/taskflow.hpp"

//#include "./taskflow/cublas_handle.hpp"
//

//-------O------------
//------/ \-----------
//-----O   O---------
//------\ /----------
//-------O------------
//-------|------------
//-------O------------
template <typename T>
void benchmark_1(size_t M, size_t N, size_t dev_id) {

  //cudaSetDevice(dev_id);

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  auto h_a = std::vector<int>(M * N, 1);
  auto h_res = std::vector<int>(M * N);

  T* dev_a1;
  T* dev_a2;
  T* dev_res;
  
  cudaMalloc(&dev_a1, M * N * sizeof(T));
  cudaMalloc(&dev_a2, M * N * sizeof(T));
  cudaMalloc(&dev_res, M * N * sizeof(T));

  auto graph = taskflow.emplace([&](tf::cudaFlowCapturer& cfc){

    auto h2d_a_t = cfc.copy(dev_a1, h_a.data(), M * N);

    auto add_1_t = cfc.transform(
      dev_a1, dev_a1 + M * N,
      [] __device__ (T& val) { return val + 1; },
      dev_a1
    );

    auto add_2_t = cfc.transform(
      dev_a2, dev_a2 + M * N,
      [] __device__ (T& val) { return val + 2; },
      dev_a1
    );

    auto multiply_t = cfc.transform(
      dev_res, dev_res + M * N,
      [] __device__ (T& lhs, T& rhs) { return lhs * rhs; },
      dev_a1, dev_a2
    );

    auto d2h_res_t = cfc.memcpy(h_res.data(), dev_res, M * N);


    h2d_a_t.precede(add_1_t).precede(add_2_t);
    add_1_t.precede(multiply_t);
    add_2_t.precede(multiply_t);
    multiply_t.precede(d2h_res_t);

  });


  executor.run(taskflow).wait();
}

int main() {
  size_t dev_id = 0;

  size_t M = 128;
  size_t N = 256;

  benchmark_1<int>(M, N, dev_id);

}
