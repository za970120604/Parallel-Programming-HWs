#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hostFE.h"
#include "helper.h"
#define GROUP_SIZE 8

void checkError(cl_int status, const char *message) {
    if (status != CL_SUCCESS) {
        fprintf(stderr, "Error: %s (Code %d)\n", message, status);
        exit(EXIT_FAILURE);
    }
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;

    // Calculate padding size and new dimensions
    int paddingSize = filterWidth >> 1;
    int paddedWidth = imageWidth + 2 * paddingSize;
    int paddedHeight = imageHeight + 2 * paddingSize;

    // Allocate and initialize the padded image
    size_t paddedImageSize = paddedWidth * paddedHeight * sizeof(float);
    float *paddedImage = (float *)calloc(paddedWidth * paddedHeight, sizeof(float));
    if (!paddedImage) {
        fprintf(stderr, "Error: Failed to allocate memory for padded image\n");
        exit(EXIT_FAILURE);
    }

    // Copy the original image into the center of the padded image
    for (int y = 0; y < imageHeight; ++y) {
        for (int x = 0; x < imageWidth; ++x) {
            paddedImage[(y + paddingSize) * paddedWidth + (x + paddingSize)] =
                inputImage[y * imageWidth + x];
        }
    }

    // Create OpenCL objects
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);
    checkError(status, "Failed to create command queue");

    int filterSize = filterWidth * filterWidth * sizeof(float);
    cl_mem device_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filterSize, filter, &status);
    checkError(status, "Failed to create buffer for filter");

    cl_mem device_input = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, paddedImageSize, paddedImage, &status);
    checkError(status, "Failed to create buffer for padded input image");

    size_t outputImageSize = imageWidth * imageHeight * sizeof(float);
    cl_mem device_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, outputImageSize, NULL, &status);
    checkError(status, "Failed to create buffer for output image");

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    checkError(status, "Failed to create kernel");

    // Set kernel arguments (add paddedWidth and originalWidth)
    status = clSetKernelArg(kernel, 0, sizeof(int), &filterWidth);
    checkError(status, "Failed to set kernel argument 0");

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_filter);
    checkError(status, "Failed to set kernel argument 1");

    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_input);
    checkError(status, "Failed to set kernel argument 2");

    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &device_output);
    checkError(status, "Failed to set kernel argument 3");

    status = clSetKernelArg(kernel, 4, sizeof(int), &paddedWidth);
    checkError(status, "Failed to set kernel argument 4");

    status = clSetKernelArg(kernel, 5, sizeof(int), &imageWidth);
    checkError(status, "Failed to set kernel argument 5");

    // Set work sizes
    size_t global_work[] = {
        ((paddedWidth + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE,
        ((paddedHeight + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE
    };
    size_t work_group[] = {GROUP_SIZE, GROUP_SIZE};

    // Enqueue kernel execution
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work, work_group, 0, NULL, NULL);
    checkError(status, "Failed to enqueue kernel");

    // Read the output buffer
    status = clEnqueueReadBuffer(command_queue, device_output, CL_TRUE, 0, outputImageSize, outputImage, 0, NULL, NULL);
    checkError(status, "Failed to read output buffer");

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseMemObject(device_filter);
    clReleaseMemObject(device_input);
    clReleaseMemObject(device_output);
    clReleaseCommandQueue(command_queue);
    free(paddedImage); // Free the padded image
}
