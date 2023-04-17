// int index = blockIdx.x * blockDim.x + threadIdx.x;

    // if (index >= cuConstRendererParams.numberOfCircles)
    //     return;

    // int index3 = 3 * index;

    // // Read position and radius
    // float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    // float  rad = cuConstRendererParams.radius[index];

    // // Compute the bounding box of the circle. The bound is in integer
    // // screen coordinates, so it's clamped to the edges of the screen.
    // short imageWidth = cuConstRendererParams.imageWidth;
    // short imageHeight = cuConstRendererParams.imageHeight;
    // short minX = static_cast<short>(imageWidth * (p.x - rad));
    // short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    // short minY = static_cast<short>(imageHeight * (p.y - rad));
    // short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // // A bunch of clamps.  Is there a CUDA built-in for this?
    // short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    // short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    // short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    // short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    // float invWidth = 1.f / imageWidth;
    // float invHeight = 1.f / imageHeight;

    // // For all pixels in the bounding box
    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
    //                                              invHeight * (static_cast<float>(pixelY) + 0.5f));
    //         shadePixel(pixelCenterNorm, p, imgPtr, index);
    //         imgPtr++;
    //     }
    // }

    // IN RENDER FUNCTIOn
    // kernelRenderCircles<<<gridDim, blockDim>>>();
    // cudaDeviceSynchronize();

// // kernelRenderCircles -- (CUDA device code)
// //
// // Each thread renders a circle.  Since there is no protection to
// // ensure order of update or mutual exclusion on the output image, the
// // resulting image will be incorrect.
// __global__ void kernelRenderPixels() {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     // parallelizing over num of pixels
//     int numberOfPixels = cuConstRendererParams.imageHeight * cuConstRendererParams.imageWidth;
//     int width = cuConstRendererParams.imageWidth;
//     int height = cuConstRendererParams.imageHeight;
    
//     if (index >= numberOfPixels)
//         return;

//     // Calculate pixel (x, y) from index
//     int pixel_x = index % width;
//     int pixel_y = index / width;
    
//     // allocate array for storing indices of circles that overlap
//     int* overlapping_indices;
//     cudaMalloc((void **)&overlapping_indices, sizeof(int) * cuConstRendererParams.numberOfCircles);
    
//     // set number of blocks
//     // set threads per block
//     int threadsPerBlock = 128;
//     int blocks = 128;
//     // kernel call to determine which circles overlaps pixel
//     find_circles(pixel_x, pixel_y, overlapping_indices);
//     // kernelCheckCircleOverlap<<<blocks, threadsPerBlock>>>(overlapping_indices, pixel_x, pixel_y);
// }




// parallelizing over circles
// __global__ void kernelCheckCircleOverlap(int* overlapping_indices, int pixel_x, int pixel_y, bool set_index) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (index >= cuConstRendererParams.numberOfCircles)
//          return;

//     // get circle position and radius from index
//     int index3 = 3 * index;

//     float3 pos = *(float3*)(&cuConstRendererParams.position[index3]);
//     float rad = cuConstRendererParams.radius[index];
//     float circle_x = pos[index3];
//     float circle_y = pos[index3+1];
//     float circle_z = pos[index3+2];

//     // check if pixel is within circle, if so, add to overlapping_indices
//     bool incircle = isPixelInCircle(pixel_x, pixel_y, circle_x, circle_y, rad);
//     if (incircle && set_index) {
//         overlapping_indices[index] = index;
//     } else if (incircle && !set_index){
//         overlapping_indices[index] = 1;
//     }
// }

// __global__ void scanKernel(  ) {
//     int linearThreadIndex =  threadIdx.y * blockDim.y + threadIdx.x;
//     __shared__ uint prefixSumInput[BLOCKSIZE];
//     __shared__ uint prefixSumOutput[BLOCKSIZE];
//     __shared__ uint prefixSumScratch[2 * BLOCKSIZE];
//     sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, prefixSumScratch, BLOCKSIZE)
// }

// static inline findCircles(int pixel_x, int pixel_y, int *overlapping_indices) {
//     int length = cuConstRendererParams.numberOfCircles;
//     int rounded_length = nextPow2(length);
    
//     int *mark_peaks;
//     cudaMalloc((void **)&mark_peaks, sizeof(int) * rounded_length);

//     // mark_peaks[i] = 1 if device_input[i] is a peak, 0 ow
//     bool set_index = false;
//     kernelCheckCircleOverlap<<<blocks, threadsPerBlock>>>(overlapping_indices, pixel_x, pixel_y, set_index);
//     cudaDeviceSynchronize();

//     // run shared memory exclusive scan
    

//     int numPeaks;
//     cudaMemcpy(&numPeaks, mark_peaks+(length - 1), sizeof(int), cudaMemcpyDeviceToHost);
    
//     int *store_peaks;
//     cudaMalloc((void **)&store_peaks, sizeof(int) * rounded_length);
    
//     set_index = true;
//     // store_peaks[i] = i if device_input[i] is a peak, 0 ow
//     kernelCheckCircleOverlap<<<blocks, threadsPerBlock>>>(overlapping_indices, pixel_x, pixel_y, set_index);
//     cudaDeviceSynchronize();
    
//     // calculate store_peaks - mark_peaks (mark_peaks is already scanned)
//     // store the subtraction output in mark_peaks
//     subtractPeaks<<<blocks, threadsPerBlock>>>(store_peaks, mark_peaks, length);
//     cudaDeviceSynchronize();

//     shiftPeaks<<<blocks, threadsPerBlock>>>(store_peaks, mark_peaks, device_output, length);
//     cudaDeviceSynchronize();

//     // then mark indices in device_output using scan
//     cudaFree(mark_peaks);
//     cudaFree(store_peaks);
// }











// old stuff before makoto saved our lives

static inline int ceilBlocks(int N, int denom)
{
    return (N + denom - 1) / denom;
}

__device__
int idx(int row, int col) {
    return row * MATRIX_WIDTH + col;
}

/* Helper function to check if pixel is in circle
 */
__device__ bool isPixelInCircle(int pixel_x, int pixel_y, float circle_x, float circle_y, float rad)
{
    // get pixel center
    float center_x = 0.5 + pixel_x;
    float center_y = 0.5 + pixel_y;

    float dist_x = center_x - circle_x;
    float dist_y = center_y - circle_y;

    // check if pixel center is in circle
    return (dist_x * dist_x + dist_y * dist_y <= rad * rad);
}

__device__ bool isValidMatrixIndex(int pixel_i, int circle_j) {
    return 0 <= pixel_i && pixel_i < cuConstRendererParams.imageHeight && 0 <= circle_j && circle_j < cuConstRendererParams.imageWidth;
}

__global__ void kernelCheckCircleOverlap(int* adj_matrix) {
    int pixel_i = blockIdx.x * blockDim.x + threadIdx.x; // pixel #
    int circle_j = blockIdx.y * blockDim.y + threadIdx.y; // circle #

    int pixel_x = pixel_i % cuConstRendererParams.imageWidth;
    int pixel_y = pixel_i / cuConstRendererParams.imageWidth;
    
    // get circle position and radius from index
    int index3 = 3 * circle_j;

    float3 pos = *(float3*)(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[circle_j];
    float circle_x = pos.x;
    float circle_y = pos.y;

    // check if pixel is within circle, if so, add to overlapping_indices
    bool incircle = isPixelInCircle(pixel_x, pixel_y, circle_x, circle_y, rad);
    if (incircle) {
        adj_matrix[idx(pixel_i,circle_j)] = 1;
    }    
}

// handle 0 index case: if indicator is 1 at 0 index, then store -1
__global__
void kernelTurnIndicatorIntoIndex(int *indicator_matrix, int *index_matrix) {
    int pixel_i = blockIdx.x * blockDim.x + threadIdx.x;
    int circle_j = blockIdx.y * blockDim.y + threadIdx.y;
    if (isValidMatrixIndex(pixel_i, circle_j)) {
        if (indicator_matrix[idx(pixel_i,circle_j)] == 1){
            if (circle_j == 0) {
                index_matrix[idx(pixel_i,circle_j)] = -1;
            } else {
                index_matrix[idx(pixel_i,circle_j)] = circle_j;
            }
        }
    }
}

__global__
void kernelSubtractIndex(int *scanned_indicator_matrix, int *index_matrix) {
    int pixel_i = blockIdx.x * blockDim.x + threadIdx.x;
    int circle_j = blockIdx.y * blockDim.y + threadIdx.y;
    if (isValidMatrixIndex(pixel_i, circle_j)) {
        index_matrix[idx(pixel_i,circle_j)] = index_matrix[idx(pixel_i,circle_j)] - scanned_indicator_matrix[idx(pixel_i,circle_j)];
    }

}

// handle 0 index case: if subtract[i][0] = -1 overlap[i][0] = 0
__global__
void kernelShiftIndex(int *shifts_matrix, int *index_matrix, int *overlap_matrix) {
    int pixel_i = blockIdx.x * blockDim.x + threadIdx.x;
    int circle_j = blockIdx.y * blockDim.y + threadIdx.y;
    if (isValidMatrixIndex(pixel_i, circle_j)) {
        if ((shifts_matrix[idx(pixel_i, circle_j)] == -1) && (circle_j == 0)) {
            overlap_matrix[idx(pixel_i, circle_j)] = 0;
        } else {
            int new_j = index_matrix[idx(pixel_i, circle_j)] - shifts_matrix[idx(pixel_i, circle_j)];
            overlap_matrix[idx(pixel_i,new_j)] = index_matrix[idx(pixel_i,new_j)];
        }
    }
}

__global__
void kernelNumCircles(int *num_circles, int *indicator_matrix) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 <= index && index < cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight) {
        int last_index_in_row = (((index + 1) * cuConstRendererParams.numberOfCircles) - 1) * sizeof(int);
        num_circles[index] = indicator_matrix[idx(index, last_index_in_row)];
    }
        
}

__device__ void renderCircleOnPixel(int pixel_i, int circle_j) {    
    int index3 = 3 * circle_j;

    // Read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);

    // Compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    int pixelX = pixel_i / cuConstRendererParams.imageWidth;
    int pixelY = pixel_i % cuConstRendererParams.imageWidth;

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
    shadePixel(pixelCenterNorm, p, imgPtr, circle_j);
}

__global__
void kernelRenderPixels(int *overlap_matrix, int *num_circles_overlap) {
    int pixel_i = blockIdx.x * blockDim.x + threadIdx.x;

    int num_overlap = num_circles_overlap[pixel_i];
    if (0 <= pixel_i && pixel_i < cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight) {
        for (int circle_j = 0; circle_j < num_overlap; circle_j++) {
            renderCircleOnPixel(pixel_i, circle_j);
        }
    }
}

// __global__
// void kernelMatrixScan(int *matrix) {
//     int linearThreadIndex =  threadIdx.y * blockDim.y + threadIdx.x;

//     __shared__ uint prefixSumInput[BLOCKSIZE];
//     __shared__ uint prefixSumOutput[BLOCKSIZE];
//     __shared__ uint prefixSumScratch[2 * BLOCKSIZE];
//     sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, prefixSumScratch, BLOCKSIZE);
// }

__global__ void
upsweep_kernel(int N, int twod, int* device_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int twod1 = twod*2;
    i = i * twod1;
    // printf("blockidx: %d blockDim.x: %d threadIdx.x: %d i: %d\n", blockIdx.x, blockDim.x, threadIdx.x, i);
    if (i+twod1-1 < N && i+twod-1 < N)
        device_data[i+twod1-1] += device_data[i+twod-1];
}

__global__ void
downsweep_kernel(int N, int twod, int* device_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int twod1 = twod*2;
    i = i * twod1;
    // printf("blockidx: %d blockDim.x: %d threadIdx.x: %d i: %d\n", blockIdx.x, blockDim.x, threadIdx.x, i);
    if (i+twod1-1 < N && i+twod-1 < N) {
        int t = device_data[i+twod-1];
        device_data[i+twod-1] = device_data[i+twod1-1];
        // change twod1 below to twod to reverse prefix sum.
        device_data[i+twod1-1] += t;
    }
}

__global__ void
last_elt_kernel(int N, int* device_data) {
    // zero out last value
    device_data[N - 1] = 0;
}


void exclusive_scan(int* device_data, int length)
{
    int threadsPerBlock = 128;

    int length_of_array = nextPow2(length);
    // cudaMemset(device_data + (length * sizeof(int)), 0, (length_of_array - length) * sizeof(int));

    // upsweep phase.
    for (int twod = 1; twod < length_of_array; twod*=2)
    {
        int twod1 = twod*2;
        int blocks = ceilBlocks(length, twod1 * threadsPerBlock);
        // int blocks = (length_of_array + threadsPerBlock - 1) / (threadsPerBlock);
        // printf("N: %d\n", N);
        // printf("blocks: %d\n", blocks);
	    upsweep_kernel<<<blocks, threadsPerBlock>>>(length, twod, device_data);
    }
    
    cudaThreadSynchronize();
    
    last_elt_kernel<<<1, 1>>>(length_of_array, device_data);
    cudaThreadSynchronize();
    // downsweep phase.
    for (int twod = length_of_array/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
        int blocks = ceilBlocks(length_of_array, twod1 * threadsPerBlock);
        // int blocks = (length_of_array + threadsPerBlock - 1) / (threadsPerBlock);
        // printf("length of array: %d\n", length_of_array);
        // printf("blocks: %d\n", blocks);
	downsweep_kernel<<<blocks, threadsPerBlock>>>(length_of_array, twod, device_data);
    }
    cudaThreadSynchronize();
}

__global__
void print_matrix(int *matrix, int row_start, int row_end, int col_start, int col_end) {
    printf("first line\n");
    printf("printing matrix from row %d to %d and col %d to %d\n", row_start, row_end, col_start, col_end);
    printf("another line");
    for (int i = row_start; i < row_end; i++) {
        for (int j = col_start; j < col_end; j++) {
            printf("hi");
            printf("matrix size: %d ", sizeof(matrix));
            printf("index: %d", i * nextPow2Device(cuConstRendererParams.numberOfCircles) + j);
            int elt = matrix[i * nextPow2Device(cuConstRendererParams.numberOfCircles) + j];
            printf("elt: %d \n", elt);
        }
        printf("|\n");
    }
    printf("end printing matrix\n");
}

void
CudaRenderer::render() {
    // // 256 threads per block is a healthy number
    // dim3 blockDim(256, 1);
    // dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // PHASE 1: parallelize over circles

    // make 2D array
    const int Nx = cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight; // number of rows
    const int Ny = nextPow2(cuConstRendererParams.numberOfCircles); // number of columns
    int num_bytes_matrix = sizeof(int) * Nx * Ny;
    printf("size of int: %d", sizeof(int));
    printf("Nx: %d", Nx);
    printf("Ny: %d", Ny);
    printf("cuConstRendererParams.imageWidth: %hu", cuConstRendererParams.imageWidth);
    printf("cuConstRendererParams.imageHeight: %hu", cuConstRendererParams.imageHeight);
    printf("num_bytes_matrix: %d", num_bytes_matrix);
    
    int *indicator_matrix;
    cudaMalloc(&indicator_matrix, num_bytes_matrix);

    int threadsPerBlock1D = 128;
    int numBlocks1D = (cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight + threadsPerBlock1D - 1)/ threadsPerBlock1D;

    dim3 threadsPerBlock(4, 4, 1);
    dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y, 1);

    // (mark_peaks)
    // this call will cause execution of 72 threads
    // (Nx*Ny)/16 blocks of 16 threads each
    // each 4 x 4 chunk of array is assigned to a block
    kernelCheckCircleOverlap<<<numBlocks, threadsPerBlock>>>(indicator_matrix);
    cudaDeviceSynchronize();

    // printf("hi\n");
    print_matrix<<<1,1>>>(indicator_matrix, 570, 580, 0, 3);
    cudaCheckError(cudaDeviceSynchronize());

    // PHASE 2: parallelize over pixels

    // scan over rows to get the number of circles that overlap each pixel
    // scan
    // kernelMatrixScan<<<SCAN_BLOCK_DIM, BLOCKSIZE>>>(indicator_matrix);
    // TODO: parallelize this later
    for (int i = 0; i < Nx; i++) {
        exclusive_scan(indicator_matrix + i * Ny, cuConstRendererParams.numberOfCircles);
    }

    // get number of circles
    int *num_circles_overlap; // 1D array
    cudaMalloc(&num_circles_overlap, sizeof(int) * cuConstRendererParams.imageWidth * cuConstRendererParams.imageHeight);
    kernelNumCircles<<<numBlocks1D, threadsPerBlock1D>>>(num_circles_overlap, indicator_matrix);

    // allocate new array for output of first scan
    int *index_matrix;
    cudaMalloc(&index_matrix, num_bytes_matrix);    

    // (store_peaks)
    // if adj_matrix[i][j] = 1 -> adj_matrix[i][j] = j (circle index)
    // exception: index 0 is represented by -1
    kernelTurnIndicatorIntoIndex<<<numBlocks, threadsPerBlock>>>(indicator_matrix, index_matrix);
    
    // (subtract peaks)
    // (store_peaks - mark_peaks)
    // index_matrix - scanned(indicator_matrix)
    kernelSubtractIndex<<<numBlocks, threadsPerBlock>>>(indicator_matrix, index_matrix);
    
    // allocate new array for output of shift
    int *overlap_matrix;
    cudaMalloc(&overlap_matrix, num_bytes_matrix);    

    // (shift peaks)
    // scan over rows again to shift relevant indices to the leftmost
    kernelShiftIndex<<<numBlocks, threadsPerBlock>>>(indicator_matrix, index_matrix, overlap_matrix);

    // parallel_for each pixel:
    // for j = 0 ... numCircles (must be sequential bc there is order)
    // render circle color on top of current pixel w maths
    // for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
    //     float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
    //     for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
    //         float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
    //                                              invHeight * (static_cast<float>(pixelY) + 0.5f));
    //         shadePixel(pixelCenterNorm, p, imgPtr, index);
    //         imgPtr++;
    //     }
    // }
    // TODO: make kernel for this for loop
    kernelRenderPixels<<<numBlocks1D, threadsPerBlock1D>>>(overlap_matrix, num_circles_overlap);
}
