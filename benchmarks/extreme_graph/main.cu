#include "../../taskflow/cudaflow.hpp"
#include "extreme_graph.hpp"

int main(int argc, char* argv[]) {
  size_t num_kernels{std::stoul(argv[1])};
  ExtremeGraph<int> graph{num_kernels};

  /*std::cout << "--------Total number of kernels: 1 + 1 + " << num_kernels*/
            /*<< "-----------------\n"*/
            /*<< "Serial:\n"*/
            /*<< "  cudaFlowCapturer: "*/
            /*<< graph.run_serial<tf::cudaFlowCapturer>(1000000000, [] __device__ (int& val){ ++val; }, 0) << " ms\n"*/
  for(size_t i = 0; i < 10; ++i)
    std::cout << graph.run_parallel<tf::cudaFlowCapturer>(100000000, [] __device__ (int& val){ ++val; }, 0) << '\n';
            /*<< "  cudaFlow: "*/
            /*<< graph.run_serial<tf::cudaFlow>(1000000000, [] __device__ (int& val){ ++val; }, 0) << " ms\n";*/

  /*std::cout << "Parallel:\n "*/
            /*<< "  cudaFlowCapturer: "*/
            /*<< graph.run_parallel<tf::cudaFlowCapturer>(1000000000, [] __device__ (int& val){ ++val; }, 0) << "ms\n"*/
            /*<< "  cudaFlow: "*/
            /*<< graph.run_parallel<tf::cudaFlow>(1000000000, [] __device__ (int& val){ ++val; }, 0) << "ms\n";*/
}
