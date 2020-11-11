#include "../taskflow/taskflow.hpp"
#include <chrono>

//---------------------------------------------------------------------------------------------------------------
//Declaration of ExtremeGraph
//---------------------------------------------------------------------------------------------------------------

template <typename T>
class ExtremeGraph {

  public:

    template <typename C>
    ExtremeGraph(size_t num_kernels, C&& callable = [] __device__ (T& val){ ++val; })

    double run(int dev_id);

  private:
    
    std::function<void(T)> _kernel;

    size_t _size;

    size_t _num_kernels;
    
};

//---------------------------------------------------------------------------------------------------------------
//Definition of ExtremeGraph
//---------------------------------------------------------------------------------------------------------------

template <typename T>
template <typename C>
ExtremeGraph::ExtremeGraph(
  size_t num_kernels,
  C&& callable
): _num_kernels(num_kernels),
   _kernel(std::forward<C>(callable))
{
}

//     O
//   /|...\
//  O O... O
//   \|.../
//     O 
template <typename T>
double ExtremeGraph::run_parallel(size_t n, int dev_id) {

  auto tic = std::chrono::steady_clock::now();

  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<T> h_vec(n, 1);

  auto dev_vec = tf::cuda_malloc_device<T>(n);

  int chunk = n / _num_kernels;
  int last_chunk; 

  (n % _num_kernels == 0) 
    ? last_chunk = chunk 
    : last_chunk = n % _num_kernels;

  taskflow.emplace_on(([&](tf::cudaFlowCapturer& cap) {

    auto h2d_t = cap.copy(dev_vec h_vec, n);

    std::vector<tf::cudaTask> kernels(_num_kernels);
    for(size_t i = 0; i < _num_kernels - 1; ++i) {
      kernels[i] = cap.for_each(
        dev_vec + i * chunk,
        dev_vec + (i + 1) * chunk,
        _kernel  
      );
    }

    kernels.back() = cap.for_each(
      dev_vec + (_num_kernels - 1) * chunk,
      dev_vec + (_num_kernels - 1) * chunk + last_chunk,
      _kernel
    );
    

    auto d2h_t = cap.copy(h_vec, dev_vec, n);

    for(auto&& kernel: kernels) {
      h2d_t.precede(kernel);
      kernel.precede(d2h_t);
    }

  }, dev_id);


  executor.run(taskflow).wait();

  auto toc = std::chrono::steady_clock::now();

  return toc - tic;
}

//O---O---O.....---O---O---O
template <typename T>
double ExtremeGraph::run_serial(size_t n, int dev_id) {

  auto tic = std::chrono::steady_clock::now();

  tf::Taskflow taskflow;
  tf::Executor executor;

  std::vector<T> h_vec(n, 1);

  auto dev_vec = tf::cuda_malloc_device<T>(n);

  int chunk = n / _num_kernels;
  int last_chunk; 

  (n % _num_kernels == 0) 
    ? last_chunk = chunk 
    : last_chunk = n % _num_kernels;

  taskflow.emplace([&](tf::cudaFlowCapturer& cap) {

    auto h2d_t = cap.copy(dev_vec h_vec, n);

    std::vector<tf::cudaTask> kernels(_num_kernels);
    for(size_t i = 0; i < _num_kernels - 1; ++i) {
      kernels[i] = cap.for_each(
        dev_vec + i * chunk,
        dev_vec + (i + 1) * chunk,
        _kernel  
      );
    }

    kernels.back() = cap.for_each(
      dev_vec + (_num_kernels - 1) * chunk,
      dev_vec + (_num_kernels - 1) * chunk + last_chunk,
      _kernel
    );
    

    auto d2h_t = cap.copy(h_vec, dev_vec, n);

    h2d_t.precede(kernels[0]);
    for(size_t i = 0; i < kernels.size() - 1; ++i) {
      kernels[i].precede(kernels[i + 1]);
    }
    kernels.back().precede(d2h_t);

  });


  executor.run(taskflow).wait();

  auto toc = std::chrono::steady_clock::now();

  return toc - tic;
}
