#include <taskflow/cudaflow.hpp>
#include <vector>
#include <chrono>

//     O
//   /|...\
//  O O... O
//   \|.../
//     O 
//
//---------------------------------------------------------------------------------------------------------------
//ParallelGraph
//---------------------------------------------------------------------------------------------------------------

class ParallelGraph: public Graph {

  public:

    ParallelGraph(int num_nodes);

    ~ParallelGraph();

};

//---------------------------------------------------------------------------------------------------------------
//Definition of ParallelGraph
//---------------------------------------------------------------------------------------------------------------

ParallelGraph::ParallelGraph(int num_nodes):
  Graph{3}
{
  _num_nodes = num_nodes;

  //graph
  std::vector<size_t> first_level_out_nodes(_num_nodes - 2);
  std::iota(
    first_level_out_nodes.begin(),
    first_level_out_nodes.end(),
    0
  );
  std::vector<Node> first_level_node(1, Node{0, 0, first_level_out_nodes});

  size_t id{0};
  std::vector<Node> second_level_nodes;
  for(int k = 0; k < _num_nodes - 2; ++k) {
    std::vector<size_t> second_level_out_nodes(1, 0);
    second_level_nodes.emplace_back(1, id++, second_level_out_nodes);
  }

  std::vector<size_t> empty;
  std::vector<Node> third_level_node(1, Node{2, 0, empty});

  _graph.resize(3);
  _graph[0] = std::move(first_level_node);
  _graph[1] = std::move(second_level_nodes);
  _graph[2] = std::move(third_level_node);

  allocate_nodes();
}

ParallelGraph::~ParallelGraph() {
  free_nodes();
}

