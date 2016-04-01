// each module gemm-XXX.cpp will have two components
//      1. setup function.
//      2. execution function.
// opencl gpu implementation of gemm.

namespace Mocha {

template <class T>
class GEMM_GPU {
public:

  static size_t size_a, size_b, size_c;  // size of the matrix.
  static size_t global_size[1];
  static size_t local_size[1];
  static cl_event event;
  static std::shared_ptr<OpenCLBasic> oclobjects;
  static std::shared_ptr<OpenCLProgramOneKernel> executable;

  static void setup() {
    err = 0;

    string cl_program = cmdparser->cl_program.getValue();
    string clkernel_path = "clkernel/";

    size_a = cmdparser->sa.getValue();
    size_b = cmdparser->sb.getValue();
    size_c = cmdparser->sc.getValue();

    local_size[0] = 256;

    if(cl_program == "blocking-2-v4") {
      clkernel_path += "gemm-blocking-2x2-vload4.cl";
      // set kernel configurations.
    }else if(cl_program == "blocking-4-v4") {
      clkernel_path += "gemm-blocking-4x4-vload4.cl";
    }else if(cl_program == "noblock-v8") {
      clkernel_path += "gemm-noblock-vload8.cl";
    }

    cout << "[cl_program] " << clkernel_path << endl;

    // build options for opencl.
    string cl_build_options =
        "-DT=" + cmdparser->arithmetic.getValue() +
        " -DT4=" + cmdparser->arithmetic.getValue() + "4" + 
        " -DT8=" + cmdparser->arithmetic.getValue() + "8" + 
        " -DT16=" + cmdparser->arithmetic.getValue() + "16" + 
        " " + (cmdparser->arithmetic_double.isSet() ? " -DSAMPLE_NEEDS_DOUBLE" : "") + 
        " " + (cmdparser->arithmetic_half.isSet() ? " -DSAMPLE_NEEDS_HALF" : "");

    cout << "Build program options: " << inquotes(cl_build_options) << "\n";

    // Build kernel
    oclobjects = std::make_shared<OpenCLBasic>(
            cmdparser->platform.getValue(),
            cmdparser->device_type.getValue(),
            cmdparser->device.getValue()
        );

    wstring clkernel_path_w;
    clkernel_path_w.assign(clkernel_path.begin(), clkernel_path.end());

    executable = std::make_shared<OpenCLProgramOneKernel>(
          *oclobjects,
          clkernel_path_w, 
          "",
          "gemm",
          cl_build_options
        );

    cout
        << "Running gemm: " << cmdparser->kernel.getValue()
        << " kernel with matrix size: " << size_a << "x" << size_b << "\t"
        << size_b << "x" << size_c << "\n";
  };


  static OpenCLDeviceAndHostMemory<T> matrix_A;
  static OpenCLDeviceAndHostMemory<T> matrix_B;
  static OpenCLDeviceAndHostMemory<T> matrix_C;
  static OpenCLDeviceAndHostMemory<int> matrix_mask;
  static int matrix_maskN;
  static cl_int err; // OpenCL error code

  static void loadMatrix(T* A, T* B, T* C, int* mask, int maskN) {
    matrix_A.host = A;
    matrix_B.host = (T*) new char[size_b * size_c * sizeof(T)];
    // transpose the matrix.
    for(size_t c = 0; c < size_c; c++) {
      for(size_t b = 0; b < size_b; b++) {
        matrix_B.host[c * size_b + b] = B[b * size_c + c];
      }
    }
    matrix_C.host = C;

    matrix_maskN = maskN;
    matrix_mask.host = mask;
    global_size[0] = (maskN + local_size[0] - 1) / local_size[0] * local_size[0];

    // load matrix into device memory.

    matrix_A.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        size_a * size_b * sizeof(T),
        matrix_A.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    matrix_B.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        size_b * size_c * sizeof(T),
        matrix_B.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    matrix_C.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        size_a * size_c * sizeof(T),
        matrix_C.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    matrix_mask.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        matrix_maskN * 2 * sizeof(int),
        matrix_mask.host,
        &err
    );
    SAMPLE_CHECK_ERRORS(err);

    err = clSetKernelArg(executable->kernel, 0, sizeof(cl_mem), &matrix_A.device);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable->kernel, 1, sizeof(cl_mem), &matrix_B.device);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable->kernel, 2, sizeof(cl_mem), &matrix_C.device);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable->kernel, 3, sizeof(cl_int), &size_a);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable->kernel, 4, sizeof(cl_int), &size_b);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable->kernel, 5, sizeof(cl_int), &size_c);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable->kernel, 6, sizeof(cl_mem), &matrix_mask.device);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(executable->kernel, 7, sizeof(cl_int), &matrix_maskN);
    SAMPLE_CHECK_ERRORS(err);


    err = clFinish(oclobjects->queue);
    SAMPLE_CHECK_ERRORS(err);
  }


  static void run(int num_iter) {
    // start execution.
    event = 0;
    for(int i = 0; i < num_iter; i++) {
      err = clEnqueueNDRangeKernel(
          oclobjects->queue,
          executable->kernel,
          1,
          0,
          global_size,
          local_size,
          0, 0, &event
      );
      SAMPLE_CHECK_ERRORS(err);
    }

    err = clFinish(oclobjects->queue);
    SAMPLE_CHECK_ERRORS(err);
  }
  

  static void loadResult() {
    clEnqueueMapBuffer(
        oclobjects->queue,
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


  static double getDeviceTime() {
    cl_ulong deviceStartTime = 0;
    cl_ulong deviceEndTime = 0;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &deviceStartTime, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &deviceEndTime, NULL);
    return deviceEndTime - deviceStartTime;
  }


  static void cleanup() {
    err = clEnqueueUnmapMemObject(
        oclobjects->queue,
        matrix_C.device,
        matrix_C.host,
        0, 0, 0
    );
    SAMPLE_CHECK_ERRORS(err);

    // Finish here is only required for correct time measurment on the next iteration
    // It does not affect correctness of calculations because you use the in-order OpenCL queue here.
    err = clFinish(oclobjects->queue);
    SAMPLE_CHECK_ERRORS(err);
  }

};

template <typename T>
size_t GEMM_GPU<T>::size_a;

template <typename T>
size_t GEMM_GPU<T>::size_b;

template <typename T>
size_t GEMM_GPU<T>::size_c;  // size of the matrix.

template <typename T>
int GEMM_GPU<T>::matrix_maskN; 

template <typename T>
size_t GEMM_GPU<T>::global_size[1];

template <typename T>
size_t GEMM_GPU<T>::local_size[1];

template <typename T>
cl_event GEMM_GPU<T>::event;

template <typename T>
std::shared_ptr<OpenCLBasic> GEMM_GPU<T>::oclobjects;

template <typename T>
std::shared_ptr<OpenCLProgramOneKernel> GEMM_GPU<T>::executable;

template <typename T>
OpenCLDeviceAndHostMemory<T> GEMM_GPU<T>::matrix_A;

template <typename T>
OpenCLDeviceAndHostMemory<T> GEMM_GPU<T>::matrix_B;

template <typename T>
OpenCLDeviceAndHostMemory<T> GEMM_GPU<T>::matrix_C;

template <typename T>
OpenCLDeviceAndHostMemory<int> GEMM_GPU<T>::matrix_mask;


template <typename T>
cl_int GEMM_GPU<T>::err; // OpenCL error code

};
