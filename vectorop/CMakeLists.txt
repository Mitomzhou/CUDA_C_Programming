cmake_minimum_required(VERSION 3.14)
project(vectorop)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
set(CU_KERNEL_ADD_FILE kernel_add.cu)
cuda_add_executable(kernel_add ${CU_KERNEL_ADD_FILE})

set(CPU_ADD_FILE cpu_add.cpp)
add_executable(cpu_add ${CPU_ADD_FILE})




