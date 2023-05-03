#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__global__ void checkOverwrite(int *darr, int length)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length)
    {
        int value = darr[idx];
        darr[idx] = idx;

        if (value != idx) {
            printf("Thread %d overwrote the value with %d\n", idx, darr[idx]);
        }
    }
}

int main()
{
    int length = 16;
    int threadsPerBlock = 4;
    int* device_data;

    cudaMalloc((void **)&device_data, sizeof(int) * length);
    int check[16];

    for (int i = 0; i < length; i++)
    {
        check[i] = i;
    }

    // cudaMemcpy(device_data, check, length * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    checkOverwrite<<<numBlocks, threadsPerBlock>>>(device_data, length);

    cudaMemcpy(check, device_data, length * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < length; i++){
        printf( "%d \n", check[i]);
    }

    // free memory
    cudaFree(device_data);

    return 0;
}
