cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(TestGEMM C CXX)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -w -std=c++11 -DCMAKE_BUILD  -fmax-errors=5")
endif()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -std=c++11")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

message("opencl includes" ${OPENCL_INCLUDES})
include_directories(${OPENCL_INCLUDES} "utils")
include_directories(${OPENCL_INCLUDES} "include")

add_executable(gemm 
  ${PROJECT_SOURCE_DIR}/src/gemm.cpp
  ${PROJECT_SOURCE_DIR}/src/cmdoptions.cpp
  ${PROJECT_SOURCE_DIR}/utils/basic.cpp
  ${PROJECT_SOURCE_DIR}/utils/cmdparser.cpp
  ${PROJECT_SOURCE_DIR}/utils/oclobject.cpp
  ${PROJECT_SOURCE_DIR}/utils/utils.cpp
  ${PROJECT_SOURCE_DIR}/utils/yuv_utils.cpp
)

target_link_libraries(gemm ${OPENCL_LIBS})

