#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication
__global__ void matrixVectorMul(const float *matrix, const float *vector, float *result, 
                              int matrixWidth, int matrixHeight) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < matrixHeight) {
        float sum = 0.0f;
        // Each thread computes one element of the result vector
        for (int col = 0; col < matrixWidth; col++) {
            sum += matrix[row * matrixWidth + col] * vector[col];
        }
        result[row] = sum;
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

// Function to print vector
void printVector(const float* vector, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.2f ", vector[i]);
    }
    printf("\n\n");
}

int main(void) {
    // Matrix dimensions
    int matrixWidth = 4;
    int matrixHeight = 4;
    size_t matrixSize = matrixWidth * matrixHeight * sizeof(float);
    size_t vectorSize = matrixWidth * sizeof(float);
    size_t resultSize = matrixHeight * sizeof(float);

    // Allocate host memory
    float *h_matrix = (float *)malloc(matrixSize);
    float *h_vector = (float *)malloc(vectorSize);
    float *h_result = (float *)malloc(resultSize);

    // Initialize input matrix and vector
    for (int i = 0; i < matrixHeight; i++) {
        for (int j = 0; j < matrixWidth; j++) {
            h_matrix[i * matrixWidth + j] = rand()/(float)RAND_MAX;
        }
    }

    for (int i = 0; i < matrixWidth; i++) {
        h_vector[i] = rand()/(float)RAND_MAX;
    }

    // Print input matrix and vector
    printf("Matrix A:\n");
    printMatrix(h_matrix, matrixWidth, matrixHeight);
    printf("Vector x:\n");
    printVector(h_vector, matrixWidth);

    // Allocate device memory
    float *d_matrix = NULL;
    float *d_vector = NULL;
    float *d_result = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_matrix, matrixSize));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_vector, vectorSize));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_result, resultSize));

    // Copy input data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_vector, h_vector, vectorSize, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (matrixHeight + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    matrixVectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_vector, d_result, 
                                                       matrixWidth, matrixHeight);

    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost));

    // Print result vector
    printf("Result vector y:\n");
    printVector(h_result, matrixHeight);

    // Verify the result (CPU computation)
    float *h_verify = (float *)malloc(resultSize);
    for (int i = 0; i < matrixHeight; i++) {
        float sum = 0.0f;
        for (int j = 0; j < matrixWidth; j++) {
            sum += h_matrix[i * matrixWidth + j] * h_vector[j];
        }
        h_verify[i] = sum;
    }

    // Check for errors
    float maxError = 0.0f;
    for (int i = 0; i < matrixHeight; i++) {
        maxError = fmax(maxError, fabs(h_result[i] - h_verify[i]));
    }
    printf("Max error: %f\n", maxError);

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_vector));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    // Free host memory
    free(h_matrix);
    free(h_vector);
    free(h_result);
    free(h_verify);

    printf("Matrix-vector multiplication completed successfully!\n");
    return 0;
} 