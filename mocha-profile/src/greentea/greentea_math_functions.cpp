/*
 * greentea_math_functions.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "greentea/greentea.hpp"
#include "greentea/greentea_math_functions.hpp"

#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"

#include "caffe/util/math_functions.hpp"

#ifdef USE_CLBLAS
#include <clBLAS.h>
#else
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#endif

// ViennaCL 1.5.1 compability fix
#ifndef VIENNACL_MINOR_VERSION
#define VIENNACL_MINOR_VERSION 5
#endif

#if VIENNACL_MINOR_VERSION > 5
#define VCL_ROW_MAJOR , true
#define VCL_COL_MAJOR , false
#else
#define VCL_ROW_MAJOR
#define VCL_COL_MAJOR
#endif

namespace caffe {


template<typename Dtype>
void greentea_gpu_gemm(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int_tp M,
                       const int_tp N, const int_tp K, const Dtype alpha,
                       const cl_mem A, const int_tp offA, const cl_mem B,
                       const int_tp offB, const Dtype beta, cl_mem C,
                       const int_tp offC) {

  viennacl::ocl::context &ctx = viennacl::ocl::current_context();
  
  std::cout << " - Device Name: " << viennacl::ocl::current_device().name() << std::endl;
  std::cout << " - context type " << viennacl::ocl::current_context().current_device().type() << CL_DEVICE_TYPE_GPU << std::endl;

  if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
    Dtype* Aptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), A, true, CL_MAP_READ,
        sizeof(Dtype) * offA, sizeof(Dtype) * M * K, 0, NULL, NULL, NULL));
    Dtype* Bptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), B, true, CL_MAP_READ,
        sizeof(Dtype) * offB, sizeof(Dtype) * N * K, 0, NULL, NULL, NULL));
    Dtype* Cptr = reinterpret_cast<Dtype*>(clEnqueueMapBuffer(
        ctx.get_queue().handle().get(), C, true, CL_MAP_READ | CL_MAP_WRITE,
        sizeof(Dtype) * offC, sizeof(Dtype) * M * N, 0, NULL, NULL, NULL));

    caffe_cpu_gemm<Dtype>(TransA, TransB, M, N, K, alpha, Aptr, Bptr, beta,
                          Cptr);

    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), A, Aptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), B, Bptr, 0, NULL,
    NULL);
    clEnqueueUnmapMemObject(ctx.get_queue().handle().get(), C, Cptr, 0, NULL,
    NULL);
  } else {
    int_tp lda = (TransA == CblasNoTrans) ? K : M;
    int_tp ldb = (TransB == CblasNoTrans) ? N : K;
    int_tp ldc = N;

#ifndef USE_CLBLAS

    typedef typename viennacl::matrix_base<Dtype,
        uint_tp, int_tp>::size_type size_type;
    typedef typename viennacl::matrix_base<Dtype,
        uint_tp, int_tp>::size_type difference_type;


    
    
    size_type A_size1 = static_cast<size_type>((TransA == CblasTrans) ? K : M);
    size_type A_size2 = static_cast<size_type>((TransA == CblasTrans) ? M : K);

    size_type B_size1 = static_cast<size_type>((TransB == CblasTrans) ? N : K);
    size_type B_size2 = static_cast<size_type>((TransB == CblasTrans) ? K : N);

      
    viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matA(A, ctx, A_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M), 
                                                       A_size2,
                                                       size_type(offA),
                                                       difference_type(1),
                                                       size_type(lda),
                                                       true);

    viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matB(B, ctx, B_size1,
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(K), B_size2,
                                                       size_type(offB),
                                                       difference_type(1),
                                                       size_type(ldb),
                                                       true);

    viennacl::matrix_base<Dtype, size_t, ptrdiff_t> matC(C, ctx, size_type(M),
                                                       size_type(0),
                                                       difference_type(1),
                                                       size_type(M),
                                                       size_type(N),
                                                       size_type(offC),
                                                       difference_type(1),
                                                       size_type(ldc),
                                                       true);



    if (TransA == CblasTrans && TransB == CblasTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), viennacl::trans(matB),
                                  matC, alpha, beta);
    else if (TransA == CblasTrans && TransB == CblasNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(matA), matB, matC, alpha,
                                  beta);
    else if (TransA == CblasNoTrans && TransB == CblasTrans)
      viennacl::linalg::prod_impl(matA, viennacl::trans(matB), matC, alpha,
                                  beta);
    else if (TransA == CblasNoTrans && TransB == CblasNoTrans)
      viennacl::linalg::prod_impl(matA, matB, matC, alpha, beta);


#else
    clblasOrder clOrder = clblasRowMajor;
    clblasTranspose clTransA =
    (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
    clblasTranspose clTransB =
    (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;

    cl_command_queue queue = ctx.get_queue().handle().get();

    if (std::is_same<Dtype, float>::value) {
      GREENTEA_CL_BLAS_CHECK(
          clblasSgemm(clOrder, clTransA, clTransB,
              M, N, K, alpha, A, offA, lda, B, offB, ldb, beta,
              C, offC, ldc, 1, &queue, 0, NULL, NULL));
    } else {
      GREENTEA_CL_BLAS_CHECK(
          clblasDgemm(clOrder, clTransA, clTransB,
              M, N, K, alpha, A, offA, lda, B, offB, ldb, beta,
              C, offC, ldc, 1, &queue, 0, NULL, NULL));
    }
#endif
  }
}

template void greentea_gpu_gemm<float>(const int_tp ctx_id,
                                       const CBLAS_TRANSPOSE TransA,
                                       const CBLAS_TRANSPOSE TransB,
                                       const int_tp M, const int_tp N,
                                       const int_tp K, const float alpha,
                                       const cl_mem A, const int_tp offA,
                                       const cl_mem B, const int_tp offB,
                                       const float beta, cl_mem C,
                                       const int_tp offC);
template void greentea_gpu_gemm<double>(const int_tp ctx_id,
                                        const CBLAS_TRANSPOSE TransA,
                                        const CBLAS_TRANSPOSE TransB,
                                        const int_tp M, const int_tp N,
                                        const int_tp K, const double alpha,
                                        const cl_mem A, const int_tp offA,
                                        const cl_mem B, const int_tp offB,
                                        const double beta, cl_mem C,
                                        const int_tp offC);

}  // namespace caffe
