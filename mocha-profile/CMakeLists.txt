cmake_minimum_required(VERSION 2.8.7)

# use ccache to speedup compilation.
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)


if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(TestGEMM C CXX)

find_package(OpenMP)
if (OPENMP_FOUND)
  message("OpenMP found")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -w -std=c++11 -DCMAKE_BUILD  -fmax-errors=5")
endif()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -std=c++11")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

message("opencl includes " ${OPENCL_INCLUDES})
message("blas includes " ${BLAS_INCLUDES} "include")
message("blas libs " ${BLAS_LIBS})

include_directories(${OPENCL_INCLUDES} "utils" "include" "${BLAS_INCLUDES}/include")

add_executable(gemm 
  ${PROJECT_SOURCE_DIR}/src/gemm.cpp
  ${PROJECT_SOURCE_DIR}/src/cmdoptions.cpp
  ${PROJECT_SOURCE_DIR}/utils/basic.cpp
  ${PROJECT_SOURCE_DIR}/utils/cmdparser.cpp
  ${PROJECT_SOURCE_DIR}/utils/oclobject.cpp
  ${PROJECT_SOURCE_DIR}/utils/utils.cpp
  ${PROJECT_SOURCE_DIR}/utils/yuv_utils.cpp

  ${PROJECT_SOURCE_DIR}/src/caffe/math_functions.cpp
)

target_link_libraries(gemm ${OPENCL_LIBS} ${BLAS_LIBS})

