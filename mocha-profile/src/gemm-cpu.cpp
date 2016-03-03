#include "caffe/util/math_functions.hpp"

namespace Mocha {

template <class T>
class GEMM_CPU {
public:

  static T* matrix_A, *matrix_B, *matrix_C;
  static size_t size_a, size_b, size_c;  // size of the matrix.

  static void setup() {
    size_a = cmdparser->sa.getValue();
    size_b = cmdparser->sb.getValue();
    size_c = cmdparser->sc.getValue();
    matrix_B = new T[size_c * size_b];
  }

  static void loadMatrix(T* A, T* B, T* C) {
    matrix_A = A;
    // transpose the matrix.
    for(size_t c = 0; c < size_c; c++) {
      for(size_t b = 0; b < size_b; b++) {
        matrix_B[c * size_b + b] = B[b * size_c + c];
      }
    }
    matrix_C = C;
  }
  
  static void run() {
    caffe::caffe_cpu_gemm<T>(CblasNoTrans, CblasTrans, 
        size_a, size_c, size_b,
        1.0, matrix_A, matrix_B, 0., matrix_C);
  }

  static void cleanup() {
    delete[] matrix_B;
  }
};


template <typename T>
size_t GEMM_CPU<T>::size_a;

template <typename T>
size_t GEMM_CPU<T>::size_b;

template <typename T>
size_t GEMM_CPU<T>::size_c;  // size of the matrix.

template <typename T>
T* GEMM_CPU<T>::matrix_A;

template <typename T>
T* GEMM_CPU<T>::matrix_B;

template <typename T>
T* GEMM_CPU<T>::matrix_C;

}
