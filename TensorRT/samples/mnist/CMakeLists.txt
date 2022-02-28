cmake_minimum_required(VERSION 3.17)
project(sampleMNIST)
set(CMAKE_CXX_STANDARD 14)

# Reference: https://blog.csdn.net/dengwanyu/article/details/86710735
set(TENSORRT_SOURCE /home/mitom/CUDALesson/TensorRT-6.0.1.5)
include_directories(${TENSORRT_SOURCE}/include)
include_directories(${TENSORRT_SOURCE}/samples/common)
set(TENSORRT_LIB_PATH "${TENSORRT_SOURCE}/lib")
set(LOGGER ${TENSORRT_SOURCE}/samples/common/logger.cpp)  # 需要单独导入
file(GLOB TENSORRT_LIB "${TENSORRT_LIB_PATH}/*.so")

# add CUDA
find_package(CUDA 10.2 REQUIRED)
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(sampleCaffeMNIST sampleCaffeMNIST.cpp ${LOGGER})
target_link_libraries(sampleCaffeMNIST ${TENSORRT_LIB} ${CUDA_LIBRARIES})

add_executable(sampleOnnxMNIST sampleOnnxMNIST.cpp ${LOGGER})
target_link_libraries(sampleOnnxMNIST ${TENSORRT_LIB} ${CUDA_LIBRARIES})
