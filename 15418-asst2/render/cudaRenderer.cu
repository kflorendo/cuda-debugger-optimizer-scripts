#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>



#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include <assert.h>

#define SCAN_BLOCK_DIM 1024

#include "exclusiveScan.cu_inl"
#include "circleBoxTest.cu_inl"

#define THREADS_PER_BLOCK 1024
#define RECTANGLE_WIDTH 32
#define RECTANGLE_HEIGHT 32
#define RECTANGLE_SIZE 1024

#define CIRCLE_CHUNK_SIZE 1023

#define MAX_CIRCLES_STORED 1024

#define NO_INDEX_PLACEHOLDER -1

#define TEST_ROW 0

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(float2 pixelCenter, float3 p, float4* imagePtr, int circleIndex) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

/* Helper function to round up to a power of 2.
 */
__device__ int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__device__ bool isValid(int circle_i)
{
    // ((circle_i >= (chunk_i * CIRCLE_CHUNK_SIZE)) && (circle_i < ((chunk_i + 1) * CIRCLE_CHUNK_SIZE))) ||
    return (circle_i < cuConstRendererParams.numberOfCircles);
}

//global_circle_indices contains circles that overlap with rectangle (rectangles by circles)
__global__ void kernelRenderPixels() {
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    int imageWidthRounded = (imageWidth % RECTANGLE_WIDTH == 0 ) ? imageWidth : imageWidth + (RECTANGLE_WIDTH - imageWidth % RECTANGLE_WIDTH);
    int imageHeightRounded = (imageHeight % RECTANGLE_WIDTH == 0 ) ? imageHeight : imageHeight + (RECTANGLE_HEIGHT - imageHeight % RECTANGLE_HEIGHT);
    int numberOfCircles = cuConstRendererParams.numberOfCircles;

    // get rectangle index and bounds
    // int rectangle_i = blockIdx.x;
    // index of the bottom left pixel of the rectangle
    // int pixel_start = rectangle_i * RECTANGLE_SIZE;

    // convert pixel indices to top left (rect_x1, rect_y1) and bottom right (rect_x2, rect_y2) of rectangle
    int rect_y1 = blockIdx.y * RECTANGLE_HEIGHT;
    int rect_x1 = blockIdx.x * RECTANGLE_WIDTH;
    int rect_y2 = min(imageHeight, rect_y1 + RECTANGLE_HEIGHT);
    int rect_x2 = min(imageWidth, rect_x1 + RECTANGLE_WIDTH);

    // create shared array of circle indices for this rectangle
    __shared__ uint circle_indices[MAX_CIRCLES_STORED];

    __shared__ int num_overlap;
    __shared__ int last_overlap;
    // printf("IN THE KERNEL\n");

    num_overlap = 0;
    last_overlap = 0;

    // iterate over circle chunks
    for (int chunk_i = 0; chunk_i < ((numberOfCircles + CIRCLE_CHUNK_SIZE - 1)/CIRCLE_CHUNK_SIZE); chunk_i++) {
        // create indicators array we'll perform scan on
        __shared__ uint circle_indicators[THREADS_PER_BLOCK];

        int circle_i = chunk_i * CIRCLE_CHUNK_SIZE + threadIdx.x;
        bool is_valid = isValid(circle_i);
        
        if (is_valid) {
            // read position and radius of circle
            int index3 = 3 * circle_i;
            float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
            float rad = cuConstRendererParams.radius[circle_i];
            // set circle indicator
            circle_indicators[threadIdx.x] = circleInBox(p.x, p.y, rad, (float)rect_x1 / (float)imageWidth, (float)rect_x2 / (float)imageWidth, (float)rect_y1 / (float)imageHeight, (float)rect_y2 / (float)imageHeight);
            if (circle_i < 3 && rect_x1 == 416 && rect_y1 == 224) {
                printf("circle %d overlap? %d", circle_i, circle_indicators[threadIdx.x]);
                printf("p.x %6.4lf\n", p.x);
                printf("p.y %6.4lf\n", p.y);
                printf("rad %6.4lf\n", rad);
                printf("rect_x1 %d rect_y1 %d rect_x2 %d rect_y2 %d\n", rect_x1, rect_y1, rect_x2, rect_y2);
                // printf("rect_x1 %6.4lf\n", ((float)rect_x1 / (float)imageWidth));
                // printf("rect_y1 %6.4lf\n", ((float)rect_y1 / (float)imageHeight));
                // printf("rect_x2 %6.4lf\n", ((float)rect_x2 / (float)imageWidth));
                printf("rect_y2 %6.4lf\n", ((float)rect_y2 / (float)imageHeight));
                // printf("p.x %6.4lf p.y %6.4lf rad %6.4lf (float)rect_x1 / (float)imageWidth %6.4lf (float)rect_x2 / (float)imageWidth %6.4lf (float)rect_y1 / (float)imageHeight %6.4lf (float)rect_y2 / (float)imageHeight %6.4lf\n", p.x, p.y, rad, ((float)rect_x1 / (float)imageWidth), ((float)rect_x2 / (float)imageWidth), ((float)rect_y1 / (float)imageHeight), ((float)rect_y2 / (float)imageHeight));
            }
                
            if (circle_i == (numberOfCircles - 1)) {
                last_overlap = circle_indicators[threadIdx.x];
            }
        }

        __syncthreads();

        // run scan on circle indicators
        int linear_thread_index = threadIdx.y * blockDim.x + threadIdx.x;
        __shared__ uint circle_indicators_scanned[THREADS_PER_BLOCK];
        __shared__ uint prefix_sum_scratch[2 * THREADS_PER_BLOCK];
        sharedMemExclusiveScan(linear_thread_index, circle_indicators, circle_indicators_scanned, prefix_sum_scratch, THREADS_PER_BLOCK);
        __syncthreads();

        if (is_valid) {
            // add last element of scan to each index to get correct index to shift to
            circle_indicators_scanned[threadIdx.x] += num_overlap;

            // set circle indices in shared array (ignore last circle)
            if (threadIdx.x < CIRCLE_CHUNK_SIZE && (circle_indicators_scanned[threadIdx.x] != circle_indicators_scanned[threadIdx.x + 1])) {
                int new_index = circle_indicators_scanned[threadIdx.x];
                circle_indices[new_index] = circle_i;
            }
        }

        // if (rect_x1 == 544 && rect_y1 == 544) {
        //     printf("circle_indicators mfor circle %d is %d at thread %d \n", circle_i, circle_indicators_scanned[threadIdx.x], threadIdx.x);
        // }
        __syncthreads();
        
        // save last element of scan
        if (threadIdx.x == 0) {
            if (chunk_i < (numberOfCircles/CIRCLE_CHUNK_SIZE)) {
                num_overlap = circle_indicators_scanned[THREADS_PER_BLOCK - 1];
            } else {
                num_overlap = circle_indicators_scanned[numberOfCircles % THREADS_PER_BLOCK - 1];
            }
        }
        __syncthreads();
        // if (rect_x1 == 575 && rect_y1 == 575) {
        //     printf("test %d \n", num_overlap);
        // }
    }

    __syncthreads();

    // printf("made it here");
    // handle if circle with last index overlaps (missed by the scan)
    if (threadIdx.x == 0) {
        num_overlap += last_overlap;
    }
    __syncthreads();

    int pixel_x = rect_x1 + (threadIdx.x % RECTANGLE_WIDTH);
    int pixel_y = rect_y1 + (threadIdx.x / RECTANGLE_WIDTH);

    // if (pixel_x == 441 && pixel_y == 230) {
    //     for (int i = 0; i < MAX_CIRCLES_STORED; i++) {
    //         printf("%d ", circle_indices[i]);
    //     }
    //     printf("rect_x1 %d rect_y1 %d rect_x2 %d rect_y2 %d\n circle_indices [%d %d %d %d]\n", rect_x1, rect_y1, rect_x2, rect_y2, circle_indices[0], circle_indices[1], circle_indices[2], circle_indices[3]);
    // }

    if (pixel_x >= imageWidth || pixel_y >= imageHeight)
        return;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // if (pixel_x == 575 && pixel_y == 575)
    //     printf("num circles overlap %d\n", num_overlap);

    
    // printf("num_overlap %d for pixel (%d, %d)\n", num_overlap, pixel_x, pixel_y);

    // shade circles over pixel sequentially
    for (int i = 0; i < num_overlap; i++){
        int circle_i = circle_indices[i];
        // if (pixel_x == 575 && pixel_y == 575)
        //     printf("shading %d over pixel (%d, %d)\n", circle_i, pixel_x, pixel_y);

        int index3 = 3 * circle_i;
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixel_x) + 0.5f),
                                        invHeight * (static_cast<float>(pixel_y) + 0.5f));
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixel_y * imageWidth + pixel_x)]);

        shadePixel(pixelCenterNorm, p, imgPtr, circle_i);
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

// Returns x rounded to the nearest multiple of y
static inline int roundMultiple(int x, int y) {
    int add = x % y;
    if (add == 0) {
        return x;
    } else {
        return x + (y - add);
    }
}

void
CudaRenderer::render() {
    // int numPixelsRounded = roundMultiple(image->width, RECTANGLE_WIDTH) * roundMultiple(image->height, RECTANGLE_HEIGHT);
    // int numRectangles = numPixelsRounded / RECTANGLE_SIZE;
    dim3 blockDim(THREADS_PER_BLOCK, 1);
    dim3 gridDim(roundMultiple(image->width, RECTANGLE_WIDTH) / RECTANGLE_WIDTH, roundMultiple(image->height, RECTANGLE_HEIGHT) / RECTANGLE_HEIGHT);
    // int numBlocks = numRectangles;

    // printf("STARTING KERNEL =====================================\n");
    kernelRenderPixels<<<gridDim, blockDim>>>();
    cudaCheckError(cudaDeviceSynchronize());
}