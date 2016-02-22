// energy profiling code for GEMM.
#ifndef MOCHA_CPP_ENERGY_PROFILE
#define MOCHA_CPP_ENERGY_PROFILE

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <tuple>


namespace Mocha {

static const char* kernel_energy_file = "/sdcard/kernel_energy.txt";

static std::pair<double, double> getCurrentEnergy() {
  std::fstream fs(kernel_energy_file, std::fstream::in);
  double time, energy;
  while(!fs.eof()) {
    std::string line;
    std::getline(fs, line);
    double _time, _energy;
    int num_read = sscanf(line.c_str(), "time %lf energy %lf", &_time, &_energy);
    if(num_read == 2) { // set only if successful.
      time = _time;
      energy = _energy;
    }
  }
  fs.close();
  return std::make_pair(time, energy);
}

}
#endif
