#include <iostream>
#include "random_DAG.hpp"
#include "cuda_executor.hpp"

int main() {

  //graph.print_graph(std::cout);
  //
  for(size_t i = 0; i < 10; ++i) {
    RandomDAG graph{32, 1024, 1024};
    cudaExecutor executor(graph, 0);

    std::cout << executor.run<tf::cudaFlowCapturer>() << '\n';
  }
  return 0;
}
