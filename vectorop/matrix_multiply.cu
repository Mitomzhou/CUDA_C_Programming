#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#define M 512
#define K 512
#define N 512

#define BLOCK_SIZE 32  //block size


/**
 * 主机端初始化matrix
 * @param array
 * @param size
 */
void initial(float* array, int size)
{
    for (int i=0; i<size; i++){
        array[i] = (float)(rand() % 10 + 1);
    }
}

/**
 * 主机端矩阵相乘 (M,K) (K,N) -> (M,N)
 * @param arrayA
 * @param arrayB
 * @param arrayC
 * @param sizeM
 * @param sizeK
 * @param sizeN
 */
void multiplicationMatrixOnHost(float* arrayA, float* arrayB, float* arrayC, int sizeM, int sizeK, int sizeN)
{
    for(int m=0; m<sizeM; m++){
        for(int n=0; n<sizeN; n++){
            float sum = 0;
            for(int k=0; k<sizeK; k++){
                sum += arrayA[sizeM*k+m] * arrayB[sizeN*k+n];
            }
            arrayC[m*sizeN+n] = sum;
        }
    }
}


__global__ void multiplicateMatrixOnDevice(float* arrayA, float* arrayB, float* arrayC, int sizeM, int sizeK, int sizeN)
{
    int ix = threadIdx.x + blockDim.x*blockIdx.x; //row number
    int iy = threadIdx.y + blockDim.y*blockIdx.y; //col number

    if(ix < sizeN && iy < sizeM){
        float sum = 0;
        for(int k=0; k<sizeK; k++){
            sum += arrayA[iy*sizeK+k] * arrayB[sizeN*k+ix];
        }
        arrayC[iy*sizeN + ix] = sum;
    }
}

int main()
{
    clock_t start = 0, finish = 0;
    float cpuRunTime;

    int Axy = M * K;
    int Bxy = K * N;
    int Cxy = M * N;

    // 主机端内存申请
    float *h_A, *h_B, *hostRef, *deviceRef;
    h_A = (float*)malloc(Axy * sizeof(float));
    h_B = (float*)malloc(Bxy * sizeof(float));
    hostRef = (float*)malloc(Cxy * sizeof(float));
    deviceRef = (float*)malloc(Cxy * sizeof(float));

    // 初始化矩阵
    initial(h_A, Axy);
    initial(h_B, Bxy);

    start = clock();
    multiplicationMatrixOnHost(h_A, h_B, hostRef, M, K, N);
    finish = clock();
    cpuRunTime = (float)(finish - start) / CLOCKS_PER_SEC;

    // 输出host用时
    std::cout << "主机端 multiplicationMatrixOnHost: " <<
                "(" << M << "," << K << ") " << "(" << K << "," << N << ") -> " << "(" << M << "," << N << ")" <<
                "Runtime: " << cpuRunTime << " (s)" << std::endl;

    // 设备端内存申请
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, Axy * sizeof(float));
    cudaMalloc((void**)&d_B, Bxy * sizeof(float));
    cudaMalloc((void**)&d_C, Cxy * sizeof(float));

    // 主机端拷贝到设备端
    cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);

    //
    int dimx = 2, dimy = 2;
    dim3 block(dimx, dimy);
    dim3 grid((M+block.x-1)/block.x, (N+block.y-1)/block.y);

    cudaEvent_t gpustart, gpustop;
    float elapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);

    multiplicateMatrixOnDevice<<<grid,block>>> (d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);
    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);

    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    // 输出Device用时
    std::cout << "设备端 multiplicateMatrixOnDevice: " <<
              "(" << M << "," << K << ") " << "(" << K << "," << N << ") -> " << "(" << M << "," << N << ")" <<
              "Runtime: " <<  elapsedTime / 1000 << " (s)" << std::endl;
    return 0;
}
