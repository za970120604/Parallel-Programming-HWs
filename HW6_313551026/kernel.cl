__kernel void convolution(const int filterWidth,
                          __constant const float *restrict filter,
                          __global const float *restrict paddedImage,
                          __global float *restrict outputImage,
                          const int paddedWidth,
                          const int originalWidth) {

    // Get global IDs
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Calculate half filter size
    int halfFilterSize = filterWidth >> 1;

    // Initialize sum for the convolution
    float sum = 0.0f;

    // Perform convolution
    for (int row_offset = -halfFilterSize; row_offset <= halfFilterSize; ++row_offset) {
        int neighborY = y + row_offset;
        int filterRowBase = (row_offset + halfFilterSize) * filterWidth;

        for (int col_offset = -halfFilterSize; col_offset <= halfFilterSize; ++col_offset) {
            int neighborX = x + col_offset;

            int imageIndex = neighborY * paddedWidth + neighborX;
            int filterIndex = filterRowBase + (col_offset + halfFilterSize);

            sum = mad(filter[filterIndex], paddedImage[imageIndex], sum);
        }
    }

    // Write to the output image (without padding)
    if (x >= halfFilterSize && x < (originalWidth + halfFilterSize) &&
        y >= halfFilterSize && y < (originalWidth + halfFilterSize)) {
        int outputX = x - halfFilterSize;
        int outputY = y - halfFilterSize;
        outputImage[outputY * originalWidth + outputX] = sum;
    }
}
