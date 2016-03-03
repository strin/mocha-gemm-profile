// each module gemm-XXX.cpp will have two components
//      1. setup function.
//      2. execution function.
// opencl gpu implementation of gemm.
#include "greentea/greentea_math_functions.hpp"

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

namespace Mocha {

template <class T>
class GEMM_VIENNACL {
public:

  static size_t size_a, size_b, size_c;  // size of the matrix.
  static size_t global_size[2];
  static size_t local_size[2];
  static cl_event event;
  static std::shared_ptr<OpenCLBasic> oclobjects;

  static void setup() {
    err = 0;

    size_a = cmdparser->sa.getValue();
    size_b = cmdparser->sb.getValue();
    size_c = cmdparser->sc.getValue();

    std::vector<viennacl::ocl::device> devices = viennacl::ocl::current_context().devices();
    viennacl::ocl::current_context().switch_device(devices[0]); // TODO: using device[0].
    viennacl::ocl::current_context().build_options("-cl-mad-enable");
  };


  static cl_int err; // OpenCL error code

  static OpenCLDeviceAndHostMemory<T> matrix_A;
  static OpenCLDeviceAndHostMemory<T> matrix_B;
  static OpenCLDeviceAndHostMemory<T> matrix_C;

  static void loadMatrix(T* A, T* B, T* C) {
    matrix_A.host = A;
    matrix_B.host = B;
    matrix_C.host = C;

    matrix_A.device = clCreateBuffer(
        viennacl::ocl::current_context().handle().get(),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        size_a * size_b * sizeof(T),
        matrix_A.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    matrix_B.device = clCreateBuffer(
        viennacl::ocl::current_context().handle().get(),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        size_b * size_c * sizeof(T),
        matrix_B.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    matrix_C.device = clCreateBuffer(
        viennacl::ocl::current_context().handle().get(),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        size_a * size_c * sizeof(T),
        matrix_C.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);
  }


  static void run() {
    caffe::greentea_gpu_gemm<T>(
        0, CblasNoTrans, CblasNoTrans,
        size_a, size_c, size_b, 1., 
        matrix_A.device, 0,
        matrix_B.device, 0,
        0.,
        matrix_C.device, 0
      );

  }


  static void loadResult() {
    clEnqueueMapBuffer(
        viennacl::ocl::current_context().get_queue().handle().get(),
        matrix_C.device,
        CL_TRUE,    // blocking map
        CL_MAP_READ,
        0,
        size_a * size_c * sizeof(T),
        0, 0, 0,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);
  }


  static void cleanup() {
  }

};

template <typename T>
size_t GEMM_VIENNACL<T>::size_a;

template <typename T>
size_t GEMM_VIENNACL<T>::size_b;

template <typename T>
size_t GEMM_VIENNACL<T>::size_c;  // size of the matrix.

template <typename T>
size_t GEMM_VIENNACL<T>::global_size[2];

template <typename T>
size_t GEMM_VIENNACL<T>::local_size[2];

template <typename T>
cl_event GEMM_VIENNACL<T>::event;

template <typename T>
std::shared_ptr<OpenCLBasic> GEMM_VIENNACL<T>::oclobjects;

template <typename T>
cl_int GEMM_VIENNACL<T>::err; // OpenCL error code

template <typename T>
OpenCLDeviceAndHostMemory<T> GEMM_VIENNACL<T>::matrix_A;

template <typename T>
OpenCLDeviceAndHostMemory<T> GEMM_VIENNACL<T>::matrix_B;

template <typename T>
OpenCLDeviceAndHostMemory<T> GEMM_VIENNACL<T>::matrix_C;

};
