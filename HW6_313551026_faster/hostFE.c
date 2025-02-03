#include <stdio.h>
#include <stdlib.h>
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
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int imageSize = imageWidth * imageHeight * sizeof(float);

    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);
    checkError(status, "Failed to create command queue");

    cl_mem device_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filterSize, filter, &status);
    checkError(status, "Failed to create buffer for filter");

    cl_mem device_input = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageSize, inputImage, &status);
    checkError(status, "Failed to create buffer for input image");

    cl_mem device_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, &status);
    checkError(status, "Failed to create buffer for output image");

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    checkError(status, "Failed to create kernel");

    status = clSetKernelArg(kernel, 0, sizeof(int), &filterWidth);
    checkError(status, "Failed to set kernel argument 0");
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_filter);
    checkError(status, "Failed to set kernel argument 1");
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_input);
    checkError(status, "Failed to set kernel argument 2");
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &device_output);
    checkError(status, "Failed to set kernel argument 3");

    size_t global_work[] = {
        ((imageWidth + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE,
        ((imageHeight + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE
    };
    size_t work_group[] = {GROUP_SIZE, GROUP_SIZE};

    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work, work_group, 0, NULL, NULL);
    checkError(status, "Failed to enqueue kernel");

    status = clEnqueueReadBuffer(command_queue, device_output, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);
    checkError(status, "Failed to read output buffer");

    clReleaseKernel(kernel);
    clReleaseMemObject(device_filter);
    clReleaseMemObject(device_input);
    clReleaseMemObject(device_output);
    clReleaseCommandQueue(command_queue);
}
