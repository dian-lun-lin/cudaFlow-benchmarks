#include <iostream>
#include <fstream>
#include <string>

class Caculator {
  
  public:
    
    Caculator(const std::string& input_file): _input_file{input_file} {}

    double compute(); // read numbers line by line, compute average

    void set_input_file(const std::string& input_file) { _input_file = input_file; };


  private:

    std::string _input_file;
};

double Caculator::compute() {
  std::ifstream f(_input_file);
  
  std::string str;
  double sum = 0;
  size_t num_lines = 0;
  
  while(getline(f, str)) {
    sum += std::stod(str);
    ++num_lines;
  }

  return sum / num_lines; 
}


int main(int argc, char* argv[]) {
  Caculator caculator(argv[1]);

  std::cout << caculator.compute() << '\n';
}
