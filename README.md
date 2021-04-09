# OPT_cudaGraph

# File Descriptions
  - main/
    - sample_graph.cu: main source file to run different graphs using cudaFlow
 
  - bencmarks/sample_graph/sample_graph/
    - extreme_graph/: graphs with sequential and embarassing parallelism 
    - map_reduce/: graph with map_reduce (i.e. Diamond)
    - random_DAG/: graph with random_DAG
    - tree/: graph with tree
    - utility/graph_executor.hpp: An executor to map a graph into taskflow. Each node contains four tasks: H2D, reduce sum, D2H, and set visited.


# Prerequisites
```
Nvidia CUDA Toolkit and Compiler (nvcc) at least v11.0 with -std=c++17
GNU C++ Compiler at least v8.0 with -std=c++17
CMake at least v3.9
```

# Step 1: Compile
Out of soucr build:
```bash
~$ mkdir build
~$ cd build
~$ cmake ../
~$ make
```

# Step 2: Run
```bash
~$ cd bin
~$ ./simple_graph -g Graph --a Args

- GraphType: SerialGraph, ParallelGraph, Tree, MapReduce, RandomDAG
- Args: 
  - SerialGraph (Sequential): number of nodes
  - ParallelGraph (Embarrassing Parallelism): number of nodes
  - Tree: degree per node, number of levels
  - MapReduce: number of partitions, number of iterations
  - RandomDAG: number of levels, maximum number of nodes per level, maximum number of edges per node
 
For example, ./simple_graph -g RandomDAG -a 10 5 8 
will launch a randomly produced DAG that has 10 levels. Each level has at most 5 nodes and each node has at most 8 edges.
```
