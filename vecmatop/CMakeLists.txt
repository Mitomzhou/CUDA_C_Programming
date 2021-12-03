cmake_minimum_required(VERSION 3.14)
project(vecmatop)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
link_directories("/usr/local/cuda-11.2/targets/x86_64-linux/lib") # libcublas.so 所在文件夹路径

set(CU_KERNEL_ADD_FILE kernel_add.cu)
cuda_add_executable(kernel_add ${CU_KERNEL_ADD_FILE})

set(CPU_ADD_FILE cpu_add.cpp)
add_executable(cpu_add ${CPU_ADD_FILE})

set(CU_MAXTRIX_MULTIPLY_FILE matrix_multiply.cu)
cuda_add_executable(matrix_multiply ${CU_MAXTRIX_MULTIPLY_FILE})
target_link_libraries(matrix_multiply cublas)


