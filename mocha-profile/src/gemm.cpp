// main profiling tool.
#include <iostream>
#include <ctime>
#include <limits>
#include <cmath>
#include <memory>

#include <CL/cl.h>

#include "cmdoptions.hpp"

using namespace std;

// global variables.
std::shared_ptr<CmdParserMochaGEMM> cmdparser;

int clGlobalSize = 1024, clLocalSize = 16;

// TODO: worry about alignment.
template <typename T>
std::tuple<T*, T*, T*> make_matrix() {
  size_t size_a = cmdparser->sa.getValue(),
         size_b = cmdparser->sb.getValue(),
         size_c = cmdparser->sc.getValue();

  size_t mem_size_a = size_a * size_b * sizeof(T);
  size_t mem_size_b = size_c * size_c * sizeof(T);
  size_t mem_size_c = size_a * size_c * sizeof(T);

  T* A = (T*) new char[mem_size_a];
  T* B = (T*) new char[mem_size_b];
  T* C = (T*) new char[mem_size_c];

  for(size_t i = 0; i < size_a; ++i)
  {
    T* row_A = A + i * size_b;
    T* row_C = C + i * size_c;

    // Fill the rows with random values from range [0, 1]
    fill_rand_uniform_01(row_A, size_b);

    // we initialize C matrix with all zeros.
    std::fill(row_C, row_C + size_c, T(0));
  }

  for(size_t i = 0; i < size_b; ++i) {
    T* row_B = B + i * size_b;

    // Fill the rows with random values from range [0, 1]
    fill_rand_uniform_01(row_B, size_c);
  }

  return std::make_tuple(A, B, C);
}

//template <typename T>
//void gemmGPU (
//    OpenCLBasic& oclobjects,
//    OpenCLProgramOneKernel& executable
//)
//{
//    // -----------------------------------------------------------------------
//    // Calculating, allocating and initializing host-side memory
//    // -----------------------------------------------------------------------
//    size_t size = cmdparser->size.getValue();
//
//    cout
//        << "Running gemm_" << cmdparser->kernel.getValue()
//        << " kernel with matrix size: " << size << "x" << size << "\n";
//
//    // Ensures that each matrix memory row is aligned
//    size_t stride = (size*sizeof(T) + rowAlignment - 1) & ~(rowAlignment - 1);
//    cout << "Memory row stride to ensure necessary alignment: " << stride << " bytes\n";
//    // calculate row stride in elements of T
//    stride /= sizeof(T);
//    assert(size <= stride);
//
//    if(stride/sizeof(T) > size_t(numeric_limits<cl_int>::max()))
//    {
//        throw Error(
//            "Memory row stride in elements " + to_str(stride/sizeof(T)) +
//            " cannot be represented as type int, which can be maximum " +
//            to_str(numeric_limits<cl_int>::max()) + "."
//        );
//    }
//
//
//    // Allocate aligned memory for matrices to use them in
//    // buffers with CL_MEM_USE_HOST_PTR.
//    // OpenCLDeviceAndHostMemory is used just for
//    // convenient resource deallocation:
//    // a pair of pointer and cl_mem object; cl_mem object is
//    // be creater later.
//
//
//    OpenCLDeviceAndHostMemory<T> matrix_A;
//    matrix_A.host = (T*)aligned_malloc(alignedSize, alignmentForPtr);
//
//    OpenCLDeviceAndHostMemory<T> matrix_B;
//    matrix_B.host = (T*)aligned_malloc(alignedSize, alignmentForPtr);
//
//    OpenCLDeviceAndHostMemory<T> matrix_C;
//    matrix_C.host = (T*)aligned_malloc(alignedSize, alignmentForPtr);
//
//    // -----------------------------------------------------------------------
//    // Allocating device-side resources for matrices
//    // -----------------------------------------------------------------------
//
//    cl_int err = 0; // OpenCL error code
//
//    // Create OpenCL buffers for the matrices based on allocated memory regions
//    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
//    // model situation when matrices are hosted by some native library that
//    // uses OpenCL to accelerate calculations.
//
//    matrix_A.device = clCreateBuffer(
//        oclobjects.context,
//        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
//        matrix_memory_size,
//        matrix_A.host,
//        &err
//    );
//    SAMPLE_CHECK_ERRORS(err);
//
//    matrix_B.device = clCreateBuffer(
//        oclobjects.context,
//        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
//        matrix_memory_size,
//        matrix_B.host,
//        &err
//    );
//    SAMPLE_CHECK_ERRORS(err);
//
//    matrix_C.device = clCreateBuffer(
//        oclobjects.context,
//        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
//        matrix_memory_size,
//        matrix_C.host,
//        &err
//    );
//    SAMPLE_CHECK_ERRORS(err);
//
//    cl_int cl_size = static_cast<int>(size);  // kernel requires int value
//    cl_int ldabc = static_cast<int>(stride);  // kernel requires int value
//
//    // extract transpose settings.
//    bool Atransposed = cmdparser->kernel_tn.isSet() or cmdparser->kernel_tt.isSet();
//    bool Btransposed = cmdparser->kernel_nt.isSet() or cmdparser->kernel_tt.isSet();
//
//    // -----------------------------------------------------------------------
//    // Define ndrange iteration space: global and local sizes based on
//    // parameters obtained from user.
//
//    // Refer to the sample documentation for clarification about
//    // how work is devided among work-groups and work-items.
//    // -----------------------------------------------------------------------
//
//    size_t global_size[2] = {
//        cmdparser->global_size.getValue(),
//        cmdparser->global_size.getValue()
//    };
//
//    size_t local_size[2] = {
//        cmdparser->local_size.getValue(),
//        cmdparser->local_size.getValue()
//    };
//
//
//    // -----------------------------------------------------------------------
//    // Setting kernel arguments
//    // -----------------------------------------------------------------------
//
//    err = clSetKernelArg(executable.kernel, 0, sizeof(cl_mem), &matrix_A.device);
//    SAMPLE_CHECK_ERRORS(err);
//    err = clSetKernelArg(executable.kernel, 1, sizeof(cl_int), &ldabc);
//    SAMPLE_CHECK_ERRORS(err);
//    err = clSetKernelArg(executable.kernel, 2, sizeof(cl_mem), &matrix_B.device);
//    SAMPLE_CHECK_ERRORS(err);
//    err = clSetKernelArg(executable.kernel, 3, sizeof(cl_int), &ldabc);
//    SAMPLE_CHECK_ERRORS(err);
//    err = clSetKernelArg(executable.kernel, 4, sizeof(cl_mem), &matrix_C.device);
//    SAMPLE_CHECK_ERRORS(err);
//    err = clSetKernelArg(executable.kernel, 5, sizeof(cl_int), &ldabc);
//    SAMPLE_CHECK_ERRORS(err);
//    err = clSetKernelArg(executable.kernel, 6, sizeof(cl_int), &cl_size);
//    SAMPLE_CHECK_ERRORS(err);
//
//    // theoretical number of floating point operations (addition and multiplication) for one kernel execution
//    // needed for performance calculations (GFLOPS) at every iteration below
//    double flops = double(size)*size*(
//        size + // multiplications
//        size   // additions
//    );
//
//    // -----------------------------------------------------------------------
//    // Loop with the kernel invocation
//    // -----------------------------------------------------------------------
//
//    for(int i = 0; i < cmdparser->iterations.getValue(); ++i)
//    {
//        // Initialize matrices row by row.
//        for(size_t i = 0; i < size; ++i)
//        {
//            T* row_A = matrix_A.host + i*stride;
//            T* row_B = matrix_B.host + i*stride;
//            T* row_C = matrix_C.host + i*stride;
//
//            // Fill the rows with random values from range [0, 1]
//            fill_rand_uniform_01(row_A, size);
//            fill_rand_uniform_01(row_B, size);
//
//            // To simplify validation a bit, we initialize C matrix with all zeros.
//            // It should not affect performance, which should be identical to
//            // the general case.
//            std::fill(row_C, row_C + size, T(0));
//        }
//
//        // Here we start measuring host time for kernel execution
//        cl_event event = 0;
//        double start = time_stamp();
//
//        err = clEnqueueNDRangeKernel(
//            oclobjects.queue,
//            executable.kernel,
//            2,
//            0,
//            global_size,
//            local_size,
//            0, 0, &event
//        );
//        SAMPLE_CHECK_ERRORS(err);
//
//        err = clFinish(oclobjects.queue);
//        SAMPLE_CHECK_ERRORS(err);
//
//        // It is important to measure end host time after clFinish call
//        double end = time_stamp();
//        double time = end - start;
//
//        cl_ulong deviceStartTime = 0;
//        cl_ulong deviceEndTime = 0;
//
//        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &deviceStartTime, NULL);
//        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &deviceEndTime, NULL);
//
//        cout << "Host time: " << time << " sec.\n";
//        cout << "Host perf: " << flops/time/1e9 << " GFLOPS\n";
//        cout << "Device perf: " << " " << flops / (deviceEndTime - deviceStartTime) << endl;
//        cout.flush();
//
//        if(i == 0 && cmdparser->validation.getValue())
//        {
//            // Validate result for the first iteration only and
//            // only if user wants this.
//            // Please note, validation procedure cannot be run at
//            // futher iterations after the very first iteration,
//            // as the results are being accumulated in C matrix
//            // every iteration but validation procedures assumes that
//            // C initial values are all zeros.
//
//            clEnqueueMapBuffer(
//                oclobjects.queue,
//                matrix_C.device,
//                CL_TRUE,    // blocking map
//                CL_MAP_READ,
//                0,
//                matrix_memory_size,
//                0, 0, 0,
//                &err
//            );
//            SAMPLE_CHECK_ERRORS(err);
//
//            // After map call, host-memory area for matrix C is
//            // automatically updated with the latest bits from the device
//            // So we just use it by original pointer as well as input matrices:
//            if(
//                !checkValidity(
//                    matrix_A.host,
//                    matrix_B.host,
//                    matrix_C.host,
//                    size,
//                    stride,
//                    Atransposed,
//                    Btransposed
//                )
//            )
//            {
//                throw Error("Validation procedure reported failures");
//            }
//
//            cout.flush();
//
//            err = clEnqueueUnmapMemObject(
//                oclobjects.queue,
//                matrix_C.device,
//                matrix_C.host,
//                0, 0, 0
//            );
//            SAMPLE_CHECK_ERRORS(err);
//
//            // Finish here is only required for correct time measurment on the next iteration
//            // It does not affect correctness of calculations because you use the in-order OpenCL queue here.
//            err = clFinish(oclobjects.queue);
//            SAMPLE_CHECK_ERRORS(err);
//        }
//    }
//
//    // All resources are deallocated automatically.
//}
//
void testGPU() { 
  // Create the necessary OpenCL objects up to device queue.
  OpenCLBasic oclobjects(
      cmdparser->platform.getValue(),
      cmdparser->device_type.getValue(),
      cmdparser->device.getValue()
  );

  // build options for opencl.
  string cl_build_options =
      "-DT=" + cmdparser->arithmetic.getValue() +
      (cmdparser->arithmetic_double.isSet() ? " -DSAMPLE_NEEDS_DOUBLE" : "");

  cout << "Build program options: " << inquotes(cl_build_options) << "\n";

  // Build kernel
  string cl_program = cmdparser->cl_program.getValue();
  wstring cl_program_w;
  cl_program_w.assign(cl_program.begin(), cl_program.end());
  OpenCLProgramOneKernel executable(
    oclobjects,
    cl_program_w, 
    "",
    "gemm",
    cl_build_options
  );

  // Call gemm with required type of elements
  //if(cmdparser->arithmetic_float.isSet())
  //{
  //  gemmGPU<float>(oclobjects, executable);
  //}
  //else if(cmdparser->arithmetic_double.isSet())
  //{
  //  gemmGPU<double>(oclobjects, executable);
  //}
}

int main(int argc, const char* argv[]) {
  try
  {
    // Define and parse command-line arguments.
    cmdparser = std::make_shared<CmdParserMochaGEMM>(argc, argv);
    cmdparser->parse();

    // Immediatly exit if user wanted to see the usage information only.
    if(cmdparser->help.isSet())
    {
      return 0;
    }

    string architecture = cmdparser->architecture.getValue();
    
    cout << "architecture " << architecture << endl;

    //if(architecture == "gpu") {
    //  test_GPU(cmdparser);
    //}

  }
  catch(const CmdParser::Error& error)
  {
      cerr
          << "[ ERROR ] In command line: " << error.what() << "\n"
          << "Run " << argv[0] << " -h for usage info.\n";
      return 1;
  }
  catch(const Error& error)
  {
      cerr << "[ ERROR ] Sample application specific error: " << error.what() << "\n";
      return 1;
  }
  catch(const exception& error)
  {
      cerr << "[ ERROR ] " << error.what() << "\n";
      return 1;
  }
  catch(...)
  {
      cerr << "[ ERROR ] Unknown/internal error happened.\n";
      return 1;
  }

  return 0;
}

