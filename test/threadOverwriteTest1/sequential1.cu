#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

int main()
{
    int length = 16;
    int check[length];

    for (int i = 0; i < length; i++)
    {
        check[i % 8] = i;
    }


    return 0;
}