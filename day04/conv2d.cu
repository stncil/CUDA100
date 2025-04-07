#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// CUDA kernel for 2D convolution
__global__ void conv2d(float *input, float *kernel, float *output, 
                      int input_width, int input_height,
                      int kernel_width, int kernel_height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < input_height && col < input_width) {
        float sum = 0.0f;
        
        // Calculate half kernel sizes for padding
        int k_half_width = kernel_width / 2;
        int k_half_height = kernel_height / 2;
        
        // Apply convolution
        for (int ky = -k_half_height; ky <= k_half_height; ky++) {
            for (int kx = -k_half_width; kx <= k_half_width; kx++) {
                int input_row = row + ky;
                int input_col = col + kx;
                
                // Handle boundary conditions with zero padding
                if (input_row >= 0 && input_row < input_height &&
                    input_col >= 0 && input_col < input_width) {
                    int kernel_row = ky + k_half_height;
                    int kernel_col = kx + k_half_width;
                    sum += input[input_row * input_width + input_col] *
                           kernel[kernel_row * kernel_width + kernel_col];
                }
            }
        }
        
        output[row * input_width + col] = sum;
    }
}

// Initialize input matrix with random values
void initialize_matrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            matrix[i * width + j] = (float)rand() / RAND_MAX;
        }
    }
}

// Print matrix
void print_matrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    // Define matrix dimensions
    const int input_width = 8;
    const int input_height = 8;
    const int kernel_width = 3;
    const int kernel_height = 3;
    
    // Allocate host memory
    float *h_input = (float *)malloc(input_width * input_height * sizeof(float));
    float *h_kernel = (float *)malloc(kernel_width * kernel_height * sizeof(float));
    float *h_output = (float *)malloc(input_width * input_height * sizeof(float));
    
    // Initialize input and kernel
    initialize_matrix(h_input, input_width, input_height);
    initialize_matrix(h_kernel, kernel_width, kernel_height);
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, input_width * input_height * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernel_width * kernel_height * sizeof(float));
    cudaMalloc((void **)&d_output, input_width * input_height * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, input_width * input_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_width * kernel_height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    conv2d<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                 input_width, input_height,
                                 kernel_width, kernel_height);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, input_width * input_height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Input Matrix:\n");
    print_matrix(h_input, input_width, input_height);
    
    printf("\nKernel Matrix:\n");
    print_matrix(h_kernel, kernel_width, kernel_height);
    
    printf("\nOutput Matrix:\n");
    print_matrix(h_output, input_width, input_height);
    
    // Free memory
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return 0;
} 