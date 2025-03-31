#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix addition
__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function to print matrix
void printMatrix(const float* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    // Matrix dimensions
    int width = 4;
    int height = 4;
    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input matrices
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            h_A[index] = rand()/(float)RAND_MAX;
            h_B[index] = rand()/(float)RAND_MAX;
        }
    }

    // Print input matrices
    printf("Matrix A:\n");
    printMatrix(h_A, width, height);
    printf("Matrix B:\n");
    printMatrix(h_B, width, height);

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

    // Copy input matrices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 threadsPerBlock(2, 2);  // 2x2 threads per block
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the CUDA kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);

    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print result matrix
    printf("Result Matrix C:\n");
    printMatrix(h_C, width, height);

    // Verify the result
    float maxError = 0.0f;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            maxError = fmax(maxError, fabs(h_C[index] - (h_A[index] + h_B[index])));
        }
    }
    printf("Max error: %f\n", maxError);

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Matrix addition completed successfully!\n");
    return 0;
} 