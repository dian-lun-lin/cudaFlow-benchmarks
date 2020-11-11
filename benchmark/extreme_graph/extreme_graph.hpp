#include "../../taskflow/cudaflow.hpp"
#include <vector>
#include <chrono>

//---------------------------------------------------------------------------------------------------------------
//Declaration of ExtremeGraph
//---------------------------------------------------------------------------------------------------------------

template <typename T>
class ExtremeGraph {

  public:

    ExtremeGraph(size_t num_kernels);

    template <typename F, typename C>
    double run_parallel(size_t n, C&& callable, int dev_id);

    template <typename F, typename C>
    double run_serial(size_t n, C&& callable, int dev_id);

  private:
    

    size_t _num_kernels;
    
};

//---------------------------------------------------------------------------------------------------------------
//Definition of ExtremeGraph
//---------------------------------------------------------------------------------------------------------------

template <typename T>
ExtremeGraph<T>::ExtremeGraph(
  size_t num_kernels
): _num_kernels(num_kernels)
{
}

//     O
//   /|...\
//  O O... O
//   \|.../
//     O 
template <typename T>
template <typename F, typename C>
double ExtremeGraph<T>::run_parallel(size_t n, C&& callable, int dev_id) {

  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<T> h_vec(n, 1);

  T* dev_vec;
  cudaMalloc(&dev_vec, sizeof(T) * n);


  int chunk = n / _num_kernels;
  int last_chunk; 

  (n % _num_kernels == 0) 
    ? last_chunk = chunk 
    : last_chunk = n % _num_kernels;

  auto cudaflow = taskflow.emplace_on([&](F& cf) {

    auto h2d_t = cf.copy(dev_vec, h_vec.data(), n);

    std::vector<tf::cudaTask> kernels(_num_kernels);
    for(size_t i = 0; i < _num_kernels - 1; ++i) {
      kernels[i] = cf.for_each(
        dev_vec + i * chunk,
        dev_vec + (i + 1) * chunk,
        callable
      );
    }

    kernels.back() = cf.for_each(
      dev_vec + (_num_kernels - 1) * chunk,
      dev_vec + (_num_kernels - 1) * chunk + last_chunk,
      callable
    );
    

    auto d2h_t = cf.copy(h_vec.data(), dev_vec, n);

    for(auto&& kernel: kernels) {
      h2d_t.precede(kernel);
      kernel.precede(d2h_t);
    }

  }, dev_id);

  auto tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

  auto toc = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
}

//O---O---O.....---O---O---O
template <typename T>
template <typename F, typename C>
double ExtremeGraph<T>::run_serial(size_t n, C&& callable, int dev_id) {

  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<T> h_vec(n, 1);

  T* dev_vec = nullptr;
  cudaMalloc(&dev_vec, sizeof(T) * n);

  int chunk = n / _num_kernels;
  int last_chunk; 

  (n % _num_kernels == 0) 
    ? last_chunk = chunk 
    : last_chunk = n % _num_kernels;

  taskflow.emplace_on([&](F& cf) {

    auto h2d_t = cf.copy(dev_vec, h_vec.data(), n);

    std::vector<tf::cudaTask> kernels(_num_kernels);
    for(size_t i = 0; i < _num_kernels - 1; ++i) {
      kernels[i] = cf.for_each(
        dev_vec + i * chunk,
        dev_vec + (i + 1) * chunk,
        callable
      );
    }

    kernels.back() = cf.for_each(
      dev_vec + (_num_kernels - 1) * chunk,
      dev_vec + (_num_kernels - 1) * chunk + last_chunk,
      callable
    );
    

    auto d2h_t = cf.copy(h_vec.data(), dev_vec, n);

    h2d_t.precede(kernels[0]);
    for(size_t i = 0; i < kernels.size() - 1; ++i) {
      kernels[i].precede(kernels[i + 1]);
    }
    kernels.back().precede(d2h_t);

  }, dev_id);


  auto tic = std::chrono::steady_clock::now();

  executor.run(taskflow).wait();

  auto toc = std::chrono::steady_clock::now();

  return std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
}
