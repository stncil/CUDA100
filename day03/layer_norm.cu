#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Constants
#define MAX_BLOCK_SIZE 1024  // Maximum threads per block
#define EPSILON 1e-5f

// Kernel to compute mean for each row
__global__ void computeRowMeanKernel(
    const float* input,
    float* mean,
    const int rows,
    const int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Shared memory for partial sums
    extern __shared__ float sdata[];
    sdata[tid] = 0.0f;
    __syncthreads();
    
    // Each thread processes multiple elements in its row
    for (int i = tid; i < cols; i += stride) {
        sdata[tid] += input[row * cols + i];
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        mean[row] = sdata[0] / cols;
    }
}

// Kernel to compute variance for each row
__global__ void computeRowVarianceKernel(
    const float* input,
    const float* mean,
    float* variance,
    const int rows,
    const int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    extern __shared__ float sdata[];
    sdata[tid] = 0.0f;
    __syncthreads();
    
    // Compute squared differences
    for (int i = tid; i < cols; i += stride) {
        float diff = input[row * cols + i] - mean[row];
        sdata[tid] += diff * diff;
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        variance[row] = sdata[0] / cols;
    }
}

// Kernel to perform layer normalization
__global__ void layerNormKernel(
    const float* input,
    float* output,
    const float* mean,
    const float* variance,
    const float* gamma,
    const float* beta,
    const int rows,
    const int cols,
    const float eps
) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (col < cols) {
        int idx = row * cols + col;
        output[idx] = gamma[col] * (input[idx] - mean[row]) / sqrt(variance[row] + eps) + beta[col];
    }
}

// Host function to perform layer normalization
void layerNorm(
    const float* input,
    float* output,
    const float* gamma,
    const float* beta,
    const int rows,
    const int cols,
    const float eps = EPSILON
) {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Calculate optimal block size
    int block_size = min(MAX_BLOCK_SIZE, prop.maxThreadsPerBlock);
    // Ensure block size is a multiple of warp size (32)
    block_size = (block_size + 31) & ~31;
    printf("Number of threads per block: %d\n", block_size);
    // Allocate device memory
    float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));
    cudaMalloc(&d_mean, rows * sizeof(float));
    cudaMalloc(&d_variance, rows * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_beta, cols * sizeof(float));
    
    // Copy input data and parameters to device
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute mean
    int num_blocks = rows;
    int shared_mem_size = block_size * sizeof(float);
    
    computeRowMeanKernel<<<num_blocks, block_size, shared_mem_size>>>(
        d_input, d_mean, rows, cols
    );
    
    // Compute variance
    computeRowVarianceKernel<<<num_blocks, block_size, shared_mem_size>>>(
        d_input, d_mean, d_variance, rows, cols
    );
    
    // Perform layer normalization
    // For normalization kernel, we need one thread per column
    int norm_block_size = min(block_size, cols);
    layerNormKernel<<<rows, norm_block_size>>>(
        d_input, d_output, d_mean, d_variance, d_gamma, d_beta,
        rows, cols, eps
    );
    
    // Copy result back to host
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

// Example usage
int main() {
    // Example dimensions
    const int rows = 2;    // 2 rows
    const int cols = 1024; // 1024 columns
    
    // Allocate host memory
    float *h_input = new float[rows * cols];
    float *h_output = new float[rows * cols];
    float *h_gamma = new float[cols];
    float *h_beta = new float[cols];
    
    // Initialize with example values
    // First row
    for (int i = 0; i < cols; i++) {
        h_input[i] = (float)(i + 1);
    }
    // Second row
    for (int i = 0; i < cols; i++) {
        h_input[cols + i] = (float)(i + 5);
    }
    
    // Initialize gamma and beta
    for (int i = 0; i < cols; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    
    // Print first few elements of input matrix
    printf("First few elements of input matrix:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", h_input[i * cols + j]);
        }
        printf("...\n");
    }
    
    // Perform layer normalization
    layerNorm(h_input, h_output, h_gamma, h_beta, rows, cols);
    
    // Print first few elements of output matrix
    printf("\nFirst few elements of output matrix after layer normalization:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", h_output[i * cols + j]);
        }
        printf("...\n");
    }
    
    // Clean up
    delete[] h_input;
    delete[] h_output;
    delete[] h_gamma;
    delete[] h_beta;
    
    return 0;
}