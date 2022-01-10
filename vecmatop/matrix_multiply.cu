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

/**
 * 设备端矩阵相乘
 * @param arrayA
 * @param arrayB
 * @param arrayC
 * @param sizeM
 * @param sizeK
 * @param sizeN
 */
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

/**
 * 设备端共享内存矩阵相乘
 * @param arrayA
 * @param arrayB
 * @param arrayC
 * @param numARows
 * @param numAColumns
 * @param numBRows
 * @param numBColumns
 * @param numCRows
 * @param numCColumns
 */
__global__ void matrixMultiplyShared(float *arrayA, float *arrayB, float *arrayC, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
    __shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float Csub = 0.0;

    for (int i = 0; i < (int)(ceil((float)numAColumns / BLOCK_SIZE)); i++)
    {
        if (i*BLOCK_SIZE + tx < numAColumns && row < numARows)
            sharedM[ty][tx] = arrayA[row*numAColumns + i * BLOCK_SIZE + tx];
        else
            sharedM[ty][tx] = 0.0;

        if (i*BLOCK_SIZE + ty < numBRows && col < numBColumns)
            sharedN[ty][tx] = arrayB[(i*BLOCK_SIZE + ty)*numBColumns + col];
        else
            sharedN[ty][tx] = 0.0;
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++)
            Csub += sharedM[ty][j] * sharedN[j][tx];
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns)
        arrayC[row*numCColumns + col] = Csub;
}

int main()
{
    clock_t start = 0, finish = 0;
    float cpuRunTime;

    int Axy = M * K;
    int Bxy = K * N;
    int Cxy = M * N;

    //***********************************
    //* 主机端矩阵相乘
    //***********************************

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
                "                        Runtime: " << cpuRunTime << " (s)" << std::endl;

    //***********************************
    //* 设备端矩阵相乘
    //***********************************

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
    cudaEventCreate(&gpustart); // 创建Event
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0); // 记录当前时间

    multiplicateMatrixOnDevice<<<grid,block>>> (d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize(); //Waits for an event to complete.
    cudaEventRecord(gpustop, 0); // 记录当前时间
    cudaEventSynchronize(gpustop); //Waits for an event to complete.Record之前的任务
    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop); //计算时间差
    cudaEventDestroy(gpustart); //destory the event
    cudaEventDestroy(gpustop);

    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    // 输出Device用时
    std::cout << "设备端 multiplicateMatrixOnDevice: " <<
              "(" << M << "," << K << ") " << "(" << K << "," << N << ") -> " << "(" << M << "," << N << ")" <<
              " <<<("<< (M+block.x-1)/block.x << "," << (N+block.y-1)/block.y << "), (" << dimx << "," << dimy << ")>>> "
              "Runtime: " <<  elapsedTime / 1000 << " (s)" << std::endl;

    //***********************************
    //* 设备端矩阵相乘（共享内存）
    //***********************************

    elapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);
    matrixMultiplyShared <<< grid, block >>> (d_A, d_B, d_C, M, K, K, N, M, N);
    cudaDeviceSynchronize();
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);

    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);

    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    // 输出Device用时
    std::cout << "设备端 matrixMultiplyShared:       " <<
              "(" << M << "," << K << ") " << "(" << K << "," << N << ") -> " << "(" << M << "," << N << ")" <<
              " <<<("<< (M+block.x-1)/block.x << "," << (N+block.y-1)/block.y << "), (" << dimx << "," << dimy << ")>>> "
              "Runtime: " <<  elapsedTime / 1000 << " (s)" << std::endl;



    //***********************************
    //* 设备端矩阵相乘(调用cublas中函数)
    //***********************************

    cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);

    elapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);

    float a = 1, b = 0;
    cublasSgemm(
            handle,
            CUBLAS_OP_T,   //矩阵A的属性参数，转置，按行优先
            CUBLAS_OP_T,   //矩阵B的属性参数，转置，按行优先
            M,          //矩阵A、C的行数
            N,          //矩阵B、C的列数
            K,          //A的列数，B的行数，此处也可为B_ROW,一样的
            &a,         //alpha的值
            d_A,        //左矩阵，为A
            K,          //A的leading dimension，此时选择转置，按行优先，则leading dimension为A的列数
            d_B,        //右矩阵，为B
            N,          //B的leading dimension，此时选择转置，按行优先，则leading dimension为B的列数
            &b,         //beta的值
            d_C,        //结果矩阵C
            M           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
    );
    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);

    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);

    // 输出Device用时
    std::cout << "设备端 cublasSgemm:                " <<
              "(" << M << "," << K << ") " << "(" << K << "," << N << ") -> " << "(" << M << "," << N << ")" <<
              " <<<("<< (M+block.x-1)/block.x << "," << (N+block.y-1)/block.y << "), (" << dimx << "," << dimy << ")>>> "
              "Runtime: " <<  elapsedTime / 1000 << " (s)" << std::endl;

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(deviceRef);

    cudaDeviceReset();

    return 0;
}
