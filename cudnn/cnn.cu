#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <cudnn.h>

__global__ void dev_const(float *px, float k) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = k;
}

__global__ void dev_iota(float *px) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    px[tid] = tid;
}

/**
 * 输出张量
 * @param data
 * @param n
 * @param c
 * @param h
 * @param w
 */
void print(const float *data, int n, int c, int h, int w) {
    std::vector<float> buffer(1 << 20);
    cudaMemcpy(buffer.data(), data,n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);
    int a = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
            for (int k = 0; k < h; ++k) {
                for (int l = 0; l < w; ++l) {
                    std::cout << std::setw(4) << std::right << buffer[a];
                    ++a;
                }
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}


int main() {

    // 句柄
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    // Input张量
    const int in_n = 1;
    const int in_c = 1;
    const int in_h = 5;
    const int in_w = 5;
    printf("Input:NCHW-(%d,%d,%d,%d)\n", in_n,in_c,in_h,in_w);

    // Input描述
    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w);

    // Input分配内存
    float *in_data;
    cudaMalloc(&in_data, in_n * in_c * in_h * in_w * sizeof(float));

    // Filter
    const int filt_k = 1;
    const int filt_c = 1;
    const int filt_h = 2;
    const int filt_w = 2;
    printf("Filter:NCHW-(%d,%d,%d,%d)\n", filt_k,filt_c,filt_h,filt_w);

    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filt_k, filt_c, filt_h, filt_w);

    float *filt_data;
    cudaMalloc(&filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float));

    // Convolution
    const int pad_h = 1;
    const int pad_w = 1;
    const int str_h = 1;
    const int str_w = 1;
    const int dil_h = 1;
    const int dil_w = 1;
    printf("Convolution:pad_h, pad_w, str_h, str_w, dil_h, dil_w: %d,%d,%d,%d,%d,%d \n", pad_h,pad_w,str_h,str_w,dil_h,dil_w);


    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc,pad_h, pad_w, str_h, str_w, dil_h, dil_w,CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    // Output
    int out_n;
    int out_c;
    int out_h;
    int out_w;

    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w);
    printf("Output:NCHW-(%d,%d,%d,%d)\n", out_n,out_c,out_h,out_w);
    std::cout << std::endl;
    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);

    float *out_data;
    cudaMalloc(&out_data, out_n * out_c * out_h * out_w * sizeof(float));


    // Algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn_handle,in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    std::cout << "Convolution algorithm: " << algo << std::endl;
    std::cout << std::endl;

    // Workspace
    size_t ws_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);

    float *ws_data;
    cudaMalloc(&ws_data, ws_size);

    std::cout << "Workspace size: " << ws_size << std::endl;
    std::cout << std::endl;

    // Perform
    float alpha = 1.f;
    float beta = 0.f;
    dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
    dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);
    cudnnConvolutionForward(cudnn_handle, &alpha, in_desc, in_data, filt_desc, filt_data,conv_desc, algo, ws_data, ws_size, &beta, out_desc, out_data);

    // Results
    std::cout << "in_data:" << std::endl;
    print(in_data, in_n, in_c, in_h, in_w);

    std::cout << "filt_data:" << std::endl;
    print(filt_data, filt_k, filt_c, filt_h, filt_w);

    std::cout << "out_data:" << std::endl;
    print(out_data, out_n, out_c, out_h, out_w);

    // Finalizing
    cudaFree(ws_data);
    cudaFree(out_data);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudaFree(filt_data);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudaFree(in_data);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroy(cudnn_handle);
    return 0;
}