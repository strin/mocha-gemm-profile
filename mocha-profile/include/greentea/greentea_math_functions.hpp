/*
 * greentea_math_functions.hpp
 *
 *  Created on: Apr 6, 2015
 *      Author: fabian
 */

#ifndef GREENTEA_MATH_FUNCTIONS_HPP_
#define GREENTEA_MATH_FUNCTIONS_HPP_

#include "caffe/definitions.hpp"

#include "caffe/util/math_functions.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/vector.hpp"

namespace caffe {



template<typename Dtype>
void greentea_gpu_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC);


}  // namespace caffe

#endif  /* GREENTEA_MATH_FUNCTIONS_HPP_ */
