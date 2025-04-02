#include <stdio.h>
#include <cuda_runtime.h>

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

// Define tile size for shared memory
#define TILE_WIDTH 32

// CUDA kernel for matrix multiplication using shared memory
__global__ void matrixMul(float *A, float *B, float *C, int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // Calculate thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate row and column of output matrix
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // Check if thread is within matrix bounds
    if (row < M && col < N) {
        // Loop over tiles
        for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
            // Load data into shared memory
            if (row < M && (i * TILE_WIDTH + tx) < K) {
                ds_A[ty][tx] = A[row * K + i * TILE_WIDTH + tx];
            } else {
                ds_A[ty][tx] = 0.0f;
            }

            if ((i * TILE_WIDTH + ty) < K && col < N) {
                ds_B[ty][tx] = B[(i * TILE_WIDTH + ty) * N + col];
            } else {
                ds_B[ty][tx] = 0.0f;
            }

            // Synchronize threads to ensure all data is loaded
            __syncthreads();

            // Compute partial sum for this tile
            for (int k = 0; k < TILE_WIDTH; k++) {
                sum += ds_A[ty][k] * ds_B[k][tx];
            }

            // Synchronize threads before loading next tile
            __syncthreads();
        }

        // Write result to output matrix
        C[row * N + col] = sum;
    }
}

// Function to initialize matrix with random values
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// Function to print matrix
void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    // A is M by K
    // B is K by N
    // C is M by N
    // C[i,j] = sum(A[i,k] * B[k,j]) for k = 0 to K-1

    // Matrix dimensions
    int M = 64;  // rows of A and C
    int K = 64;  // cols of A, rows of B
    int N = 64;  // cols of B and C

    // Calculate sizes
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    // Initialize input matrices
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    // Print input matrices
    printf("Matrix A:\n");
    printMatrix(h_A, M, K);
    printf("Matrix B:\n");
    printMatrix(h_B, K, N);

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, sizeA));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, sizeB));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, sizeC));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Print result matrix
    printf("Result Matrix C:\n");
    printMatrix(h_C, M, N);

    // Verify the result (CPU computation)
    float *h_verify = (float *)malloc(sizeC);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_verify[i * N + j] = sum;
        }
    }

    // Check for errors
    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        maxError = fmax(maxError, fabs(h_C[i] - h_verify[i]));
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
    free(h_verify);

    printf("Matrix multiplication completed successfully!\n");
    return 0;
} 