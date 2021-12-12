cmake_minimum_required(VERSION 3.14)
project(cudnn)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
link_directories("/usr/local/cuda-11.2/targets/x86_64-linux/lib")

set(CU_CNN_FILE cnn.cu)
cuda_add_executable(cnn ${CU_CNN_FILE})
target_link_libraries(cnn cudnn)


