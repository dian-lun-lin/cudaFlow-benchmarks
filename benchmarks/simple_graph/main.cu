#include <utility/CLI11/CLI11.hpp>
#include <taskflow/cudaflow.hpp>
#include "graph_executor.hpp"
#include "graph_declaration.hpp"
#include <iostream>

int main(int argc, char* argv[]) {

  CLI::App app{"Simple graph"};

  std::string mode{"SerialGraph"};
  app.add_option(
    "-m, --mode", 
    mode, 
    "select mode(SerialGraph, ParallelGraph, Tree, RandomDAG, Diamond(map-reduce)), default is SerialGraph" 
  );

  std::vector<int> args;
  app.add_option(
    "-a, --args",
    args,
    "args for constructing a graph"
  );

  CLI11_PARSE(app, argc, argv);

  Graph* g_ptr;
  if(mode == "SerialGraph") {
    assert(args.size() == 1);
    g_ptr = new SerialGraph(args[0]);
  }
  else if(mode == "ParallelGraph") {
    assert(args.size() == 1);
    g_ptr = new ParallelGraph(args[0]);
  }
  else if(mode == "Tree") {
    assert(args.size() == 2);
    g_ptr = new Tree(args[0], args[1]);
  }
  else if(mode == "RandomDAG") {
    assert(args.size() == 3);
    g_ptr = new RandomDAG(args[0], args[1], args[2]);
  }
  else if(mode == "Diamond") {
    assert(args.size() == 2);
    g_ptr = new Diamond(args[0], args[1]);
  }
  else {
    throw std::runtime_error("No such graph\n");
  }


  GraphExecutor<tf::cudaFlow> executor(*g_ptr, 0); 
  auto time_pair = executor.run();

  std::cout << "Construction time: " 
            << time_pair.first
            << " ms\n"
            << "Execution time: "
            << time_pair.second
            << " ms\n";

  g_ptr->print_graph(std::cout);
}
