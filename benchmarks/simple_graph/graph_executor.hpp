#pragma once

#include "./graph_base.hpp"
#include <taskflow/cudaflow.hpp>
#include <chrono>
#include <cassert>

template <typename CF>
class GraphExecutor {

  public:
  
    GraphExecutor(Graph& graph, int dev_id = 0);

    double run();

  private:
    
    int _dev_id;

    Graph& _g;

};

template <typename CF>
GraphExecutor<CF>::GraphExecutor(Graph& graph, int dev_id): _g{graph}, _dev_id{dev_id} {
  //TODO: why we cannot put cuda lambda function here?
}

template <typename CF>
double GraphExecutor<CF>::run() {
  tf::Taskflow taskflow;
  tf::Executor executor;

  auto trav_t = taskflow.emplace_on([this](CF& cf) {
    std::vector<std::vector<tf::cudaTask>> tasks;
    tasks.resize(_g.get_graph().size());

    for(size_t l = 0; l < _g.get_graph().size(); ++l) {
      tasks[l].resize((_g.get_graph())[l].size());
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        bool* v = _g.at(l, i).visited;
        tasks[l][i] = cf.single_task([v] __device__ () {
          *v = true;
        });
      }
    }

    for(size_t l = 0; l < _g.get_graph().size() - 1; ++l) {
      for(size_t i = 0; i < (_g.get_graph())[l].size(); ++i) {
        for(auto&& out_node: _g.at(l, i).out_nodes) {
          tasks[l][i].precede(tasks[l + 1][out_node]);
        }
      }
    }

  }, _dev_id).name("traverse");

  auto check_t = taskflow.emplace([this](){
    assert(_g.traversed());
  });
  
  trav_t.precede(check_t);

  auto tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

  auto toc = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();

}

