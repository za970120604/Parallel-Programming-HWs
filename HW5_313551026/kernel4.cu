#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_STREAM 18

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX, int resY, int offset, int maxIterations, int* device_img) {
    int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
    int y_dim = blockIdx.y * blockDim.y + threadIdx.y + offset;

    if (x_dim >= resX || y_dim >= resY) {
        return;
    }

    float2 c = make_float2(lowerX + x_dim * stepX, lowerY + y_dim * stepY);
	float2 z = c;
	float2 new_z;

	int i;
	for (i = 0; i < maxIterations; ++i) {

		if (z.x * z.x + z.y * z.y > 4.f)
			break;

		new_z.x = z.x * z.x - z.y * z.y;
		new_z.y = 2.f * z.x * z.y;
		z.x = c.x + new_z.x;
		z.y = c.y + new_z.y;
	}

    device_img[y_dim * resX + x_dim] = i;
}


void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *host_img = (int*) malloc(resX * resY * sizeof(int));
    int *device_img;
    cudaMalloc(&device_img, resX * resY * sizeof(int));
    
    cudaStream_t streams[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    int row_per_stream = resY / NUM_STREAM;
    int remainder = resY % NUM_STREAM;

    dim3 block(16, 16);  // Block size
    int offset = 0;
    for (int i = 0; i < NUM_STREAM; i++) {
        int current_rows = (i == NUM_STREAM - 1) ? row_per_stream + remainder : row_per_stream;
        dim3 grid((resX + block.x - 1) / block.x, (current_rows + block.y - 1) / block.y);

        mandelKernel<<<grid, block, 0, streams[i]>>>(lowerX, lowerY, stepX, stepY, resX, resY, offset, maxIterations, device_img);
        offset += current_rows;
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_STREAM; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaMemcpy(host_img, device_img, resX * resY * sizeof(int), cudaMemcpyDefault);
    cudaFree(device_img);
    memcpy(img, host_img, resX * resY * sizeof(int));
    free(host_img);
}
