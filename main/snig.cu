#include <SNIG/SNIG.hpp>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/scoring.hpp>

#include <CLI11/CLI11.hpp>

#include <iostream>

int main(int argc, char* argv[]) {

  //  ***All files should be converted to binary first***

  // usage: 
  //        --weight(-w)                 :  path of weight directory
  //        --input(-i)                  :  path of input file
  //        --golden(-g)                 :  path of golden file
  //        --num_neurons(-n)            :  number of neurons 1024, 4096, 16384, or 65536
  //        --num_layers(-l)             :  number of layers 120, 480, or 1920
  //        --bias(-b)                   :  bias
  //        --num_gpus                   :  number of GPUs 1, 2, 3, 4, ...
  //        --input_batch_size           :  input batch size, must be a factor of num_inputs (60000)
  //        --num_weight_buffers         :  number of weight buffers, must be an even number
  //        --thread_dimension           :  thread dimsion for inference kernel, constrained by the maximum number of threads (typically 1024)

  //example1:  
  //        ./snig

  //example2:  
  //        ./snig  -w ../sample_data/weight/neuron1024/ -i ../sample_data/MNIST/sparse-images-1024.b -g ../sample_data/MNIST/neuron1024-l120-categories.b -n 1024 -l 120 -b -0.3 --num_gpus 1 --input_batch_size 5000 --num_weight_buffers 2 --thread_dimension 2 512 1

  CLI::App app{"SNIG"};

  std::fs::path weight_path("../benchmarks/IEEE_Graph_Challenge/sample_data/weight/neuron1024/");
  app.add_option(
    "-w, --weight",
    weight_path,
    "weight directory path"
  )->check(CLI::ExistingDirectory);

  std::fs::path input_path("../benchmarks/IEEE_Graph_Challenge/sample_data/MNIST/sparse-images-1024.b");
  app.add_option(
      "-i, --input",
      input_path, 
      "input binary file path, default is ../benchmarks/IEEE_Graph_Challenge/sample_data/MNIST/sparse-images-1024.b"
  )->check(CLI::ExistingFile);

  std::fs::path golden_path("../benchmarks/IEEE_Graph_Challenge/sample_data/MNIST/neuron1024-l120-categories.b");
  app.add_option(
      "-g, --golden",
      golden_path, 
      "golden binary file path, default is ../benchmarks/IEEE_Graph_Challenge/sample_data/MINIST/neuron1024-l120-categories.b"
  );
  

  size_t num_neurons = 1024;
  app.add_option(
    "-n, --num_neurons", 
    num_neurons, 
    "total number of neurons, default is 1024"
  );

  size_t num_layers = 120;
  app.add_option(
    "-l, --num_layers",
    num_layers, 
    "total number of layers, default is 120"
  );

  float bias = -0.3f;
  app.add_option(
    "-b, --bias",
    bias,
    "bias, default is -0.3"
  );

  size_t num_gpus = 1;
  app.add_option(
    "--num_gpus", 
    num_gpus,
    "number of GPUs, default is 1"
  );
  
  size_t num_weight_buffers = 2;
  app.add_option(
    "--num_weight_buffers", 
    num_weight_buffers,
    "number of weight buffers, default is 2, must be an even number"
  );
  
  size_t input_batch_size = 5000;
  app.add_option(
    "--input_batch_size", 
    input_batch_size,
    "number of input bath size, default is 5000, must be a factor of num_input (60000)"
  );

  //for kernel dimesion
  //default is (2, 512, 1)
  std::vector<size_t> thread_vector(3);
  thread_vector[0] = 2;
  thread_vector[1] = 512;
  thread_vector[2] = 1;

  app.add_option(
    "-t, --thread_dimension",
    thread_vector,
    "thread dimension for inference kernel, need 3 parameters, default is 2 512 1, constrained by the maximum number of threads (typically 1024)"
  )->expected(3);

  CLI11_PARSE(app, argc, argv);

  Eigen::Matrix<int, Eigen::Dynamic, 1> cudaflow_result;
  //Eigen::Matrix<int, Eigen::Dynamic, 1> cudacapturer_result;

  dim3 thread_dimension{thread_vector[0], thread_vector[1], thread_vector[2]};

  snig::SNIG<float> snig(
    thread_dimension,
    weight_path, 
    bias,
    num_neurons, 
    num_layers
  );

  cudaflow_result = snig.infer<tf::cudaFlow>(input_path, 60000, input_batch_size, num_weight_buffers, num_gpus);
  //cudacapturer_result = snig.infer<tf::cudaFlowCapturer>(input_path, 60000, input_batch_size, num_weight_buffers, num_gpus);

  auto golden = snig::read_golden_binary(golden_path);

  if(snig::is_passed(cudaflow_result, golden)) {
    std::cout << "CHALLENGE PASSED (cudaflow)\n";
  }
  else{
    std::cout << "CHALLENGE FAILED (cudaflow)\n";
  }

  //if(snig::is_passed(cudacapturer_result, golden)) {
    //std::cout << "CHALLENGE PASSED (cuda_capturer)\n";
  //}
  //else{
    //std::cout << "CHALLENGE FAILED (cuda_capturer)\n";
  //}

  return 0;
}
