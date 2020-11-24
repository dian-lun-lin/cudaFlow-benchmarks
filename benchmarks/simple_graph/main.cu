#include <taskflow/cudaflow.hpp>
#include "graph_executor.hpp"
#include "graph_declaration.hpp"
#include <iostream>

int main() {
 
  RandomDAG g(3, 3, 4);
  GraphExecutor<tf::cudaFlow> executor(g, 0); 
  executor.run();
  g.print_graph(std::cout);

}
