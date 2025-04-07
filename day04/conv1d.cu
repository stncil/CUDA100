#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Constants
#define MAX_BLOCK_SIZE 1024
#define MAX_THREADS_PER_BLOCK 1024

// Kernel for 1D convolution
__global__ void conv1dKernel(
    const float* input,    // Input signal
    const float* kernel,   // Convolution kernel
    float* output,         // Output signal
    const int input_size,  // Size of input signal
    const int kernel_size, // Size of kernel
    const int output_size  // Size of output signal
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within output bounds
    if (idx < output_size) {
        float sum = 0.0f;
        
        // Perform convolution
        for (int k = 0; k < kernel_size; k++) {
            int input_idx = idx + k;
            if (input_idx < input_size) {
                sum += input[input_idx] * kernel[kernel_size - 1 - k];
            }
        }
        
        // Store result
        output[idx] = sum;
    }
}

// Function to calculate optimal grid and block dimensions
void calculateOptimalGridAndBlock(
    int output_size,
    int& grid_size,
    int& block_size
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Calculate optimal block size
    block_size = min(MAX_BLOCK_SIZE, prop.maxThreadsPerBlock);
    // Ensure block size is a multiple of warp size (32) and doesn't exceed device limits
    block_size = min((block_size + 31) & ~31, prop.maxThreadsPerBlock);
    
    // Calculate grid size
    grid_size = (output_size + block_size - 1) / block_size;
    
    printf("Device Properties:\n");
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Optimized Configuration:\n");
    printf("  Block size: %d\n", block_size);
    printf("  Grid size: %d\n", grid_size);
}

// Host function to perform 1D convolution
void conv1d(
    const float* input,
    const float* kernel,
    float* output,
    const int input_size,
    const int kernel_size
) {
    // Calculate output size
    int output_size = input_size - kernel_size + 1;
    
    // Calculate optimal grid and block dimensions
    int grid_size, block_size;
    calculateOptimalGridAndBlock(output_size, grid_size, block_size);
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    
    // Copy input data and kernel to device
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    conv1dKernel<<<grid_size, block_size>>>(
        d_input, d_kernel, d_output,
        input_size, kernel_size, output_size
    );
    
    // Copy result back to host
    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

// Example usage
int main() {
    // Example signal and kernel
    const int input_size = 10;
    const int kernel_size = 3;
    
    float input[input_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float kernel[kernel_size] = {0.5f, 1.0f, 0.5f};  // Simple smoothing kernel
    
    // Calculate output size
    int output_size = input_size - kernel_size + 1;
    float output[output_size];
    
    // Print input signal
    printf("Input signal:\n");
    for (int i = 0; i < input_size; i++) {
        printf("%.2f ", input[i]);
    }
    printf("\n\n");
    
    // Print kernel
    printf("Kernel:\n");
    for (int i = 0; i < kernel_size; i++) {
        printf("%.2f ", kernel[i]);
    }
    printf("\n\n");
    
    // Perform convolution
    conv1d(input, kernel, output, input_size, kernel_size);
    
    // Print output signal
    printf("Output signal after convolution:\n");
    for (int i = 0; i < output_size; i++) {
        printf("%.2f ", output[i]);
    }
    printf("\n");
    
    return 0;
} 