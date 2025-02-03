__kernel void convolution(const int filterWidth,
                          __constant const float *restrict filter,
                          __global const float *restrict inputImage,
                          __global float *restrict outputImage) {

    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);
    int halfFilterSize = filterWidth >> 1;

    int x = get_global_id(0);
    int y = get_global_id(1);

    float sum = 0.0f;

    if (x >= halfFilterSize && x < imageWidth - halfFilterSize &&
        y >= halfFilterSize && y < imageHeight - halfFilterSize) {
        
        for (int row_offset = -halfFilterSize; row_offset <= halfFilterSize; row_offset++) {
            int neighborY = y + row_offset;
            int filterRowBase = (row_offset + halfFilterSize) * filterWidth;

            for (int col_offset = -halfFilterSize; col_offset <= halfFilterSize; col_offset++) {
                int neighborX = x + col_offset;

                int imageIndex = neighborY * imageWidth + neighborX;
                int filterIndex = filterRowBase + (col_offset + halfFilterSize);

                sum = mad(filter[filterIndex], inputImage[imageIndex], sum);
            }
        }
    }

    outputImage[y * imageWidth + x] = sum;
}
