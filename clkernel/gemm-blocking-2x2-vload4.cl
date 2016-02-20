#ifdef SAMPLE_NEEDS_DOUBLE
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif
#ifdef SAMPLE_NEEDS_HALF
    #pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

#define DOT(a,b) \
    (a.S0 * b.S0 + a.S1 * b.S1 + a.S2 * b.S2 + a.S3 * b.S3 \
    +a.S4 * b.S4 + a.S5 * b.S5 + a.S6 * b.S6 + a.S7 * b.S7) 

#define SUM(a) \
    (a.S0 + a.S1 + a.S2 + a.S3 + a.S4 + a.S5 + a.S6 + a.S7)

__kernel void gemm (
    __global const T * restrict A,
    __global const T * restrict B,
    __global T * restrict C,
    int size_a, 
    int size_b, 
    int size_c
)
{
    const int i = get_global_id(0) * 2;
    const int j = get_global_id(1) * 2;
    
    T4 ab = (T4)0.0f;

    for (int l = 0; l < size_b; l += 4)
    {
        T4 a0 = vload4(0, &A[i * size_b]);
        T4 a1 = vload4(0, &A[(i+1) * size_b]);
        T4 b0 = vload4(0, &B[j * size_b]);
        T4 b1 = vload4(0, &B[(j+1) * size_b]);

        ab += ( T4 ) ( dot (a0 , b0 ), dot (a0 , b1 ), dot (a1 , b0 ), dot (a1 , b1 ));
        
        A += 4; 
        B += 4;
    }

    vstore2(ab.s01, 0, &C[i * size_c + j]);
    vstore2(ab.s23, 0, &C[(i+1) * size_c + j]);
}