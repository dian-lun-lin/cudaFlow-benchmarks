#include "../../taskflow/cudaflow.hpp"
#include "extreme_graph.hpp"

int main() {
  ExtremeGraph<int> graph{64};

  std::cout << graph.run_serial<tf::cudaFlowCapturer>(10000000, [] __device__ (int& val){ ++val; }, 0) << "ms\n";
  std::cout << graph.run_serial<tf::cudaFlow>(10000000, [] __device__ (int& val){ ++val; }, 0) << "ms\n";
}
