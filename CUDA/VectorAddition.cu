#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>

#define SIZE 1024

__global__ void VectorAddition(int*a, int*b, int*c)
{
    int i = threadIdx.x;

    if (i < SIZE)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int* MatrixA, *MatrixB, *MatrixC;

    cudaMallocManaged(&MatrixA, SIZE * sizeof(int));
    cudaMallocManaged(&MatrixB, SIZE * sizeof(int));
    cudaMallocManaged(&MatrixC, SIZE * sizeof(int));

    cudaDeviceSynchronize();

    for (int i = 0; i < SIZE; ++i)
    {
        MatrixA[i] = (2 * i);
        MatrixB[i] = ((2 * i) + 1);
        MatrixC[i] = 0;
    }

    VectorAddition<<<1, SIZE>>> (MatrixA, MatrixB, MatrixC);

    for (int i = 0; i < 10; i++)
    {
        printf("c[&d] = %d\n");
    }

    cudaFree(MatrixA);
    cudaFree(MatrixB);
    cudaFree(MatrixC);
}
