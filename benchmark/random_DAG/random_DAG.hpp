#pragma once

#include <random>
#include <algorithm>
#include <vector>
#include <cstring>

class cudaExecutor;
class RandomDAG;

//-------------------------------------------------------------------------
//Node
//-------------------------------------------------------------------------

class Node {

  friend cudaExecutor;
  friend RandomDAG;

  public:

    Node(
      size_t level, size_t idx,
      std::vector<size_t>& out_nodes
    );

    inline void mark() { *_visited = true; }

    inline void unmark() { *_visited = false; }

    inline bool is_visited() { return *_visited; }

    inline void print_node(std::ostream& os);


  private:

    size_t _level;
    size_t _idx;
    bool* _visited{nullptr}; //allocated by cudaMallocManaged

    std::vector<size_t> _out_nodes; 
};

Node::Node(
  size_t level, size_t idx,
  std::vector<size_t>& out_nodes
)
: _level{level}, _idx{idx},
  _out_nodes{std::move(out_nodes)}
{
}

void Node::print_node(std::ostream& os) {
  os << "id: " << _idx << " out_nodes: ";
  for(auto&& node: _out_nodes) {
    os << node << ' ';
  }
  os << "\nStatus: " << *_visited;
  os << '\n';
}

//-------------------------------------------------------------------------
//RandomDAG
//-------------------------------------------------------------------------

class RandomDAG {

  friend cudaExecutor;

  public:

    RandomDAG(
      size_t level, size_t max_nodes_per_level
    );


    inline size_t num_nodes() { return _num_nodes; }

    inline void print_graph(std::ostream& os);

    inline bool traversed();

  private:
    
    size_t _level;
    size_t _max_nodes_per_level;
    size_t _max_edges_per_node{8};

    size_t _num_nodes{0};

    std::vector<std::vector<Node>> _graph;
};

RandomDAG::RandomDAG(
  size_t level, size_t max_nodes_per_level
)
: _level{level}, _max_nodes_per_level{max_nodes_per_level}
{

  std::random_device device;
  std::mt19937 gen(device());
  std::srand(0);
  std::uniform_int_distribution<int> dist(1, max_nodes_per_level);

  size_t cur_num_nodes = 1; // root
  for(size_t l = 0; l < _level; ++l) {
    std::vector<Node> cur_nodes;
    cur_nodes.reserve(cur_num_nodes); // number of nodes at current level

    size_t next_num_nodes = dist(gen); //number of nodes at next level

    std::vector<int> next_level_nodes(next_num_nodes);
    std::iota(next_level_nodes.begin(), next_level_nodes.end(), 0);

    //create edges for each node
    for(size_t i = 0; i < cur_num_nodes; ++i) {
      if(l != _level - 1) {
        std::shuffle(next_level_nodes.begin(), next_level_nodes.end(), gen);
        size_t edges = std::rand() % _max_edges_per_node + 1;
        if(edges > next_num_nodes) {
          edges = next_num_nodes;
        }

        std::vector<size_t> out_nodes(
          next_level_nodes.begin(),
          next_level_nodes.begin() + edges
        );

        cur_nodes.emplace_back(l, i, out_nodes);
      }
      else {
        std::vector<size_t> empty;
        cur_nodes.emplace_back(l, i, empty);
      }
    }

    _graph.emplace_back(std::move(cur_nodes));

    _num_nodes += cur_num_nodes;

    cur_num_nodes = next_num_nodes;
  }

  bool* visited_ptr;
  cudaMallocManaged(&visited_ptr, sizeof(bool) * _num_nodes);
  std::memset(visited_ptr, 0, sizeof(bool) * _num_nodes);

  for(size_t l = 0; l < _level; ++l) {
    for(size_t i = 0; i < _graph[l].size(); ++i) {
      _graph[l][i]._visited = visited_ptr++;
    }
  }
}

void RandomDAG::print_graph(std::ostream& os) {
  size_t l{0};
  for(auto&& nodes: _graph) {
    os << "-----------Level: " << l++ << "-------------\n";
    for(auto&& node: nodes) {
      node.print_node(os);
    }
  }
}

bool RandomDAG::traversed() {
  for(auto&& nodes: _graph) {
    for(auto&& node: nodes) {
      if(!node.is_visited()) {
        return false;
      }
    }
  }
  return true;
}
