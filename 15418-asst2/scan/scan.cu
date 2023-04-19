#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"


extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
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

/* Helper function to round up
 */
static inline int ceilBlocks(int N, int denom)
{
    return (N + denom - 1) / denom;
}

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

__global__ void
upsweep_kernel(int N, int twod, int* device_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int twod1 = twod*2;
    i = i * twod1;
    // printf("blockidx: %d blockDim.x: %d threadIdx.x: %d i: %d\n", blockIdx.x, blockDim.x, threadIdx.x, i);
    if (i+twod1-1 < N && i+twod-1 < N) {
        device_data[i+twod1-1] += device_data[i+twod-1];
    }
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
    /* TODO
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */
    int N = length;
    int threadsPerBlock = 16;

    // int nextPow2_N = nextPow2(N);
    int length_of_array = nextPow2(N);
    // int length_of_array = (N == nextPow2_N) ? N : nextPow2_N;
    cudaMemset(device_data + (N * sizeof(int)), 0, (length_of_array - N) * sizeof(int));

    // upsweep phase.
    for (int twod = 1; twod < length_of_array; twod*=2)
    {
        int twod1 = twod*2;
        int blocks = ceilBlocks(N, twod1 * threadsPerBlock);
        // int blocks = (length_of_array + threadsPerBlock - 1) / (threadsPerBlock);
        // printf("N: %d\n", N);
        // printf("blocks: %d\n", blocks);
	    upsweep_kernel<<<blocks, threadsPerBlock>>>(N, twod, device_data);
    }
    
    cudaCheckError(cudaThreadSynchronize());
    
    last_elt_kernel<<<1, 1>>>(length_of_array, device_data);
    cudaCheckError(cudaThreadSynchronize());
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
    cudaCheckError(cudaThreadSynchronize());
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__
void markPeaks(int *device_input, int *device_output, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (0 < index && index < length - 1) {
        // printf("index: %d\n", index);
        // printf("length: %d\n", length);
        // if peak, store element at the same index in device_output
        if (device_input[index] > device_input[index + 1] && device_input[index] > device_input[index - 1]) {
            device_output[index] = 1;
        }
    }
}

__global__
void print_device_arr(int length, int *mark_peaks) {
    printf("printing mark peaks");
    for (int i = 0; i < length; i++) {
        printf("%d ", mark_peaks[i]);
    }
    printf("\n");
}

__global__
void storePeaks(int *device_input, int *store_peaks, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < index && index < length - 1) {
        // if peak, store element at the same index in device_output
        if (device_input[index] > device_input[index + 1] && device_input[index] > device_input[index - 1]) {
            store_peaks[index] = index;
        }
    }
}

__global__
void subtractPeaks(int *store_peaks, int *mark_peaks, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < index && index < length - 1) {
        mark_peaks[index] = store_peaks[index] - mark_peaks[index];
    }
}

__global__
void shiftPeaks(int *store_peaks, int *mark_peaks, int *device_output, int length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < index && index < length - 1 && mark_peaks[index] > 0) {
        int new_index = store_peaks[index] - mark_peaks[index];
        device_output[new_index] = store_peaks[index];
    }
}


int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */
    // first find peaks normally
    int blocks = length;
    int threadsPerBlock = 1;
    int rounded_length = nextPow2(length);
    
    int *mark_peaks;
    cudaMalloc((void **)&mark_peaks, sizeof(int) * rounded_length);

    // mark_peaks[i] = 1 if device_input[i] is a peak, 0 ow
    markPeaks<<<blocks, threadsPerBlock>>>(device_input, mark_peaks, length);
    cudaCheckError(cudaDeviceSynchronize());
    exclusive_scan(mark_peaks, length);

    int mark_peaks_host[rounded_length];
    cudaMemcpy(mark_peaks_host, mark_peaks, rounded_length * sizeof(int),
               cudaMemcpyDeviceToHost);

    // print_device_arr<<<1,1>>>(length, mark_peaks);

    int numPeaks = mark_peaks_host[length - 1];
    // for (int i = 0; i < length; i++) {
    //     printf("%d ", mark_peaks_host[i]);
    // }
    // printf("\n");

    // printf("numpeaks: %d\n", numPeaks);

    
    int *store_peaks;
    cudaMalloc((void **)&store_peaks, sizeof(int) * rounded_length);

    // store_peaks[i] = i if device_input[i] is a peak, 0 ow
    storePeaks<<<blocks, threadsPerBlock>>>(device_input, store_peaks, length);
    cudaCheckError(cudaDeviceSynchronize());
    
    // calculate store_peaks - mark_peaks (mark_peaks is already scanned)
    // store the subtraction output in mark_peaks
    subtractPeaks<<<blocks, threadsPerBlock>>>(store_peaks, mark_peaks, length);
    cudaCheckError(cudaDeviceSynchronize());

    shiftPeaks<<<blocks, threadsPerBlock>>>(store_peaks, mark_peaks, device_output, length);
    cudaCheckError(cudaDeviceSynchronize());

    // 1 3 2 5 4 1 7 1
    // 0 1 0 1 0 0 1 0 // markPeaks
    // 0 0 1 1 2 2 2 3 // scan on markPeaks

    // 0 1 0 3 0 0 6 0 // storePeaks
    // 0 0 1 1 2 2 2 3 // scan on markPeaks
    // _ 1 _ 2 _ _ 4 _ // storePeaks - (scan on markPeaks) = shifts
    // 1 3 6 0 0 0 0 0 // shift storePeaks[i] to the left by shifts[i]

    // 0 1 0 2 0 0 4 0 // shifts to get to the right place
    // 0 0 1 1 4 4 4 10
    // 0 1 -1 2 -4 -4 2 -10
    // 1 3 0 0 0 0 // want

    // then mark indices in device_output using scan
    return numPeaks;
    // 1 5 2 4 3
    // peak is 5, 4
    // 0 1 0 1 0 - markPeaks
    // 0 0 1 1 2 - scan on markPeaks
    // 0 1 0 3 0 - storePeaks
    // 0 1 -1 2 -2
}



/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}