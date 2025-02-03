#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX,int  maxIterations, int* device_img) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

    float c_re = lowerX + x_dim * stepX;
    float c_im = lowerY + y_dim * stepY;

    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIterations; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    
    device_img[y_dim * resX + x_dim] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *host_img = (int*) malloc(resX * resY * sizeof(int));
    int *device_img;
    cudaMalloc(&device_img, resX * resY * sizeof(int));

    dim3 block(16, 16);
    dim3 grid(resX / block.x, resY / block.y);
    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, resX, maxIterations, device_img);

    cudaMemcpy(host_img, device_img, resX * resY * sizeof(int), cudaMemcpyDefault);
    cudaFree(device_img);
    memcpy(img, host_img, resX * resY * sizeof(int));
    free(host_img);
}
