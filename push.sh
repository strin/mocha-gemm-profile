if [ -z $2 ]
then
  TEST_PATH=/sdcard/blas
else
  TEST_PATH=$2
fi

adb push mocha-profile/build/gemm $TEST_PATH/gemm-sparse

adb push clkernel/gemm-blocking-2x2-vload4.cl $TEST_PATH/clkernel/gemm-blocking-2x2-vload4.cl
adb push clkernel/gemm-blocking-4x4-vload4.cl $TEST_PATH/clkernel/gemm-blocking-4x4-vload4.cl
adb push clkernel/gemm-noblock-vload8.cl $TEST_PATH/clkernel/gemm-noblock-vload8.cl
#if [ -z $1 ]
#then
#  adb push gemm.cl $TEST_PATH/gemm.cl
#else
#  adb push $1 $TEST_PATH/gemm.cl
#fi
