#include <iostream>
#include "random_DAG.hpp"
#include "cuda_executor.hpp"

int main() {

  RandomDAG graph{5, 32};
  //graph.print_graph(std::cout);
  //
  cudaExecutor executor(graph, 0);

  std::cout << executor.run<tf::cudaFlow>();

  return 0;
}
