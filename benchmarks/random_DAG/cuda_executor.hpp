#pragma once

#include "random_DAG.hpp"
#include "../../taskflow/cudaflow.hpp"
#include <chrono>
#include <cassert>

class cudaExecutor {

  public:
  
    cudaExecutor(RandomDAG& graph, int dev_id = 0);

    template <typename CF>
    double run();

  private:
    
    int _dev_id;

    RandomDAG& _dag;
};

cudaExecutor::cudaExecutor(RandomDAG& dag, int dev_id): _dag(dag), _dev_id{dev_id} {
}

template <typename CF>
double cudaExecutor::run() {

  auto tic = std::chrono::steady_clock::now();

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto trav_t = taskflow.emplace_on([this](CF& cf) {
    std::vector<std::vector<tf::cudaTask>> tasks;
    tasks.resize(_dag._graph.size());

    for(size_t l = 0; l < _dag._graph.size(); ++l) {
      tasks[l].resize(_dag._graph[l].size());
      for(size_t i = 0; i < _dag._graph[l].size(); ++i) {
        bool* v = _dag._graph[l][i]._visited;
        tasks[l][i] = cf.single_task([v] __device__ () {
          *v = true;
        });
      }
    }

    for(size_t l = 0; l < _dag._graph.size() - 1; ++l) {
      for(size_t i = 0; i < _dag._graph[l].size(); ++i) {
        for(auto&& out_node: _dag._graph[l][i]._out_nodes) {
          tasks[l][i].precede(tasks[l + 1][out_node]);
        }
      }
    }

  }, _dev_id).name("traverse");

  auto check_t = taskflow.emplace([this](){
    assert(_dag.traversed());
  });
  
  trav_t.precede(check_t);

  executor.run(taskflow).wait();
  auto toc = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();

}


