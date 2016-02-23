#ifdef SAMPLE_NEEDS_DOUBLE
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#ifdef SAMPLE_NEEDS_HALF
    #pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

__kernel void gemm (
    __global const T * restrict A,
    __global const T * restrict B,
    __global T * restrict C,
    int size_a, 
    int size_b, 
    int size_c
)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    T8 sum = (T8)0.0f;
  
    A += i * size_b;
    B += j * size_b;

    for (int l = 0; l < size_b; l += 8)
    {
        T8 x = vload8(0, A);
        T8 y = vload8(0, B);

        sum += x * y;

        A += 8; // this is faster than A[i* k + l]. 11 GFlops vs. 10 GFlops.
        B += 8;
    }

    C[i * size_c + j] = sum.S0 + sum.S1 + sum.S2 + sum.S3
                                  + sum.S4 + sum.S5 + sum.S6 + sum.S7;
}

