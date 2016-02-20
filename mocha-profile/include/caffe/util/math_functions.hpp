#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "caffe/definitions.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template<typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                    const int_tp M, const int_tp N, const int_tp K,
                    const Dtype alpha, const Dtype* A, const Dtype* B,
                    const Dtype beta, Dtype* C);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
