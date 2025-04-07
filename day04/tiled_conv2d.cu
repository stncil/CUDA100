#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + 2)  // Extra space for halo elements

// CUDA kernel for tiled 2D convolution
__global__ void tiled_conv2d(float *input, float *kernel, float *output,
                            int input_width, int input_height,
                            int kernel_width, int kernel_height) {
    // Shared memory for input tile
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    
    // Calculate thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    // Calculate half kernel sizes for padding
    int k_half_width = kernel_width / 2;
    int k_half_height = kernel_height / 2;
    
    // Each thread loads one element into shared memory
    // First load the main block elements
    if (row < input_height && col < input_width) {
        input_tile[ty + k_half_height][tx + k_half_width] = 
            input[row * input_width + col];
    } else {
        input_tile[ty + k_half_height][tx + k_half_width] = 0.0f;
    }
    
    // Load halo elements
    // Left halo
    if (tx < k_half_width) {
        int halo_col = col - k_half_width + tx;
        if (halo_col >= 0 && row < input_height) {
            input_tile[ty + k_half_height][tx] = 
                input[row * input_width + halo_col];
        } else {
            input_tile[ty + k_half_height][tx] = 0.0f;
        }
    }
    
    // Right halo
    if (tx >= BLOCK_SIZE - k_half_width) {
        int halo_col = col + (tx - (BLOCK_SIZE - k_half_width)) + 1;
        if (halo_col < input_width && row < input_height) {
            input_tile[ty + k_half_height][tx + 2 * k_half_width] = 
                input[row * input_width + halo_col];
        } else {
            input_tile[ty + k_half_height][tx + 2 * k_half_width] = 0.0f;
        }
    }
    
    // Top halo
    if (ty < k_half_height) {
        int halo_row = row - k_half_height + ty;
        if (halo_row >= 0 && col < input_width) {
            input_tile[ty][tx + k_half_width] = 
                input[halo_row * input_width + col];
        } else {
            input_tile[ty][tx + k_half_width] = 0.0f;
        }
    }
    
    // Bottom halo
    if (ty >= BLOCK_SIZE - k_half_height) {
        int halo_row = row + (ty - (BLOCK_SIZE - k_half_height)) + 1;
        if (halo_row < input_height && col < input_width) {
            input_tile[ty + 2 * k_half_height][tx + k_half_width] = 
                input[halo_row * input_width + col];
        } else {
            input_tile[ty + 2 * k_half_height][tx + k_half_width] = 0.0f;
        }
    }
    
    // Corner halos
    if (tx < k_half_width && ty < k_half_height) {
        int halo_row = row - k_half_height + ty;
        int halo_col = col - k_half_width + tx;
        if (halo_row >= 0 && halo_col >= 0) {
            input_tile[ty][tx] = input[halo_row * input_width + halo_col];
        } else {
            input_tile[ty][tx] = 0.0f;
        }
    }
    
    // Synchronize to ensure all threads have loaded their data
    __syncthreads();
    
    // Only compute output for valid threads
    if (row < input_height && col < input_width) {
        float sum = 0.0f;
        
        // Apply convolution using shared memory
        for (int ky = -k_half_height; ky <= k_half_height; ky++) {
            for (int kx = -k_half_width; kx <= k_half_width; kx++) {
                int kernel_row = ky + k_half_height;
                int kernel_col = kx + k_half_width;
                sum += input_tile[ty + k_half_height + ky][tx + k_half_width + kx] *
                       kernel[kernel_row * kernel_width + kernel_col];
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
    tiled_conv2d<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
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