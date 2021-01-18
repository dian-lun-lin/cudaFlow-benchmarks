#pragma once

#include <simple_graph/base/graph_base.hpp>
#include <taskflow/cudaflow.hpp>
#include <taskflow/taskflow.hpp>
#include <chrono>
#include <cassert>

template <typename CF>
class GraphExecutor {

  public:
  
    GraphExecutor(Graph& graph, int dev_id = 0);

    std::pair<double, double> run();

  private:
    
    int _dev_id;

    Graph& _g;

};

template <typename CF>
GraphExecutor<CF>::GraphExecutor(Graph& graph, int dev_id): _g{graph}, _dev_id{dev_id} {
  //TODO: why we cannot put cuda lambda function here?
}

template <typename CF>
std::pair<double, double> GraphExecutor<CF>::run() {

  auto constr_tic = std::chrono::steady_clock::now();

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto trav_t = taskflow.emplace_on([this](tf::cudaFlowCapturer& cf) {
    cf.make_optimizer<tf::cudaRoundRobinCapturing>(2);
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

  auto constr_toc = std::chrono::steady_clock::now();

  auto exec_tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

  auto exec_toc = std::chrono::steady_clock::now();

  auto constr_dur = std::chrono::duration_cast<std::chrono::milliseconds>(constr_toc - constr_tic).count();

  auto exec_dur = std::chrono::duration_cast<std::chrono::milliseconds>(exec_toc - exec_tic).count();


  return {constr_dur, exec_dur};

}

