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
) {
    const int i = get_global_id(0) * 4;
    const int j = get_global_id(1) * 4;
    
    T16 sum = (T16)0.0f;

    for (int l = 0; l < size_b; l += 4)
    {
        T8 a01 = (T8) (vload4(0, &A[i * size_b]), vload4(0, &A[(i+1) * size_b]));
        T8 a23 = (T8) (vload4(0, &A[(i+2) * size_b]), vload4(0, &A[(i+3) * size_b]));
        T8 b01 = (T8) (vload4(0, &B[j * size_b]), vload4(0, &B[(j+1) * size_b]));
        T8 b23 = (T8) (vload4(0, &B[(j+2) * size_b]), vload4(0, &B[(j+3) * size_b]));

        sum += (T16) (dot(a01.lo, b01.lo), dot(a01.lo, b01.hi), dot(a01.lo, b23.lo), dot(a01.lo, b23.hi),
                          dot(a01.hi, b01.lo), dot(a01.hi, b01.hi), dot(a01.hi, b23.lo), dot(a01.hi, b23.hi),
                          dot(a23.lo, b01.lo), dot(a23.lo, b01.hi), dot(a23.lo, b23.lo), dot(a23.lo, b23.hi),
                          dot(a23.hi, b01.lo), dot(a23.hi, b01.hi), dot(a23.hi, b23.lo), dot(a23.hi, b23.hi));
        
        A += 4; 
        B += 4;
    }

    vstore4(sum.lo.lo, 0, &C[i * size_c + j]);
    vstore4(sum.lo.hi, 0, &C[(i + 1) * size_c + j]);
    vstore4(sum.hi.lo, 0, &C[(i + 2) * size_c + j]);
    vstore4(sum.hi.hi, 0, &C[(i + 3) * size_c + j]);
}