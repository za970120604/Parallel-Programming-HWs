#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX,int  maxIterations, int* device_img, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int x_dim = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

    int i[4];
    for (int j = 0; j < 4; j++) {
        float c_re = lowerX + (x_dim + j) * stepX;
        float c_im = lowerY + y_dim * stepY;
        float z_re = c_re;
        float z_im = c_im;
        for (i[j] = 0; i[j] < maxIterations; ++i[j]) {
            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }

    *((int4 *)((int *)((void *)device_img + y_dim * pitch) + x_dim)) = *((int4 *) i);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *host_img;
    int *device_img;
    size_t pitch;
    cudaHostAlloc(&host_img, resX * resY * sizeof(int), cudaHostAllocDefault);
    cudaMallocPitch(&device_img, &pitch, resX * sizeof(int), resY);

    dim3 block(16, 16);
    dim3 grid(resX / block.x / 4, resY / block.y);
    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, resX, maxIterations, device_img, pitch);

    cudaMemcpy2D(host_img, resX * sizeof(int), device_img, pitch, resX * sizeof(int), resY, cudaMemcpyDefault);
    cudaFree(device_img);
    memcpy(img, host_img, resX * resY * sizeof(int));
    cudaFreeHost(host_img);
}