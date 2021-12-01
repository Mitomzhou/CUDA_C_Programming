//
// Created by mitom on 12/1/21.
//
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

__global__
void vecAddKernel(float* A_d, float* B_d, float* C_d, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) C_d[i] = A_d[i] + B_d[i];
}

int main() {

    int n = 10000000;
    cout << n << endl;

    // 定义分配内存大小
    size_t size = n * sizeof(float);

    // host memery
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    float *da = NULL;
    float *db = NULL;
    float *dc = NULL;

    // device memery
    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&db, size);
    cudaMalloc((void **)&dc, size);

    // host->device拷贝
    cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,c,size,cudaMemcpyHostToDevice);

    // 计时结构体
    struct timeval t1, t2;

    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1)/threadPerBlock;  // 类似ceil()，节省内存.
    printf("threadPerBlock: %d \nblockPerGrid: %d \n",threadPerBlock,blockPerGrid);

    gettimeofday(&t1, NULL);

    vecAddKernel <<< blockPerGrid, threadPerBlock >>> (da, db, dc, n);

    gettimeofday(&t2, NULL);

    // device->host结果拷贝
    cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);

    double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << timeuse << endl;

    // 释放内存
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);
    return 0;
}
