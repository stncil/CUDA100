#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        printf("CUDA Runtime Error at: %s:%d\n", file, line);
        printf("%s %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}

// Matrix dimensions
const int M = 4096;  // Rows of A
const int N = 4096;  // Columns of B
const int K = 4096;  // Columns of A / Rows of B

// Block dimensions for Tensor Core operations
const int BLOCK_SIZE = 16;  // Tensor Cores work with 16x16x16 blocks
const int CHUNK_K = 16;     // Size of K dimension chunks

// Kernel using Tensor Cores
__global__ void matmul_tensorcore(half* A, half* B, float* C) {
    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Shared memory for matrix tiles
    __shared__ half As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float result[BLOCK_SIZE][BLOCK_SIZE];  // Shared memory for result
    
    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    // Calculate the base row and column for this thread block
    int warp_row = blockIdx.y * BLOCK_SIZE;
    int warp_col = blockIdx.x * BLOCK_SIZE;
    
    // Ensure we're not accessing out of bounds
    if (warp_row >= M || warp_col >= N) return;
    
    // Loop over the K dimension in chunks
    for (int i = 0; i < K; i += CHUNK_K) {
        // Load the current chunk of A and B into shared memory
        if (threadIdx.y < BLOCK_SIZE && threadIdx.x < BLOCK_SIZE) {
            // Load A tile
            if (warp_row + threadIdx.y < M && i + threadIdx.x < K) {
                As[threadIdx.y][threadIdx.x] = A[(warp_row + threadIdx.y) * K + i + threadIdx.x];
            } else {
                As[threadIdx.y][threadIdx.x] = __float2half(0.0f);
            }
            
            // Load B tile
            if (i + threadIdx.y < K && warp_col + threadIdx.x < N) {
                Bs[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + warp_col + threadIdx.x];
            } else {
                Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Load the fragments from shared memory
        nvcuda::wmma::load_matrix_sync(a_frag, &As[0][0], BLOCK_SIZE);
        nvcuda::wmma::load_matrix_sync(b_frag, &Bs[0][0], BLOCK_SIZE);
        
        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
    }
    
    // Store the result to shared memory
    if (warp_row < M && warp_col < N) {
        nvcuda::wmma::store_matrix_sync(&result[0][0], c_frag, BLOCK_SIZE, nvcuda::wmma::mem_row_major);
        
        __syncthreads();
        
        // Store with FP32 precision
        if (threadIdx.y < BLOCK_SIZE && threadIdx.x < BLOCK_SIZE) {
            if (warp_row + threadIdx.y < M && warp_col + threadIdx.x < N) {
                C[(warp_row + threadIdx.y) * N + (warp_col + threadIdx.x)] = result[threadIdx.y][threadIdx.x];
            }
        }
    }
}

// Shared memory based matrix multiplication kernel
__global__ void matmul_shared(half* A, half* B, float* C) {
    // Shared memory for matrix tiles
    __shared__ half As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Calculate row and column of C element to process
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Initialize accumulator to zero
    float sum = 0.0f;
    
    // Loop over tiles of the input matrices
    for (int tile = 0; tile < K; tile += BLOCK_SIZE) {
        // Load tile of A into shared memory
        if (row < M && (tile + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        // Load tile of B into shared memory
        if ((tile + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        if (row < M && col < N) {
            // Use FP32 for intermediate calculations to match tensor core
            float partial_sum = 0.0f;
            for (int k = 0; k < BLOCK_SIZE && (tile + k) < K; k++) {
                float a_val = __half2float(As[threadIdx.y][k]);
                float b_val = __half2float(Bs[k][threadIdx.x]);
                partial_sum += a_val * b_val;
            }
            sum += partial_sum;  // Accumulate in FP32
        }
        
        // Synchronize before loading the next tile
        __syncthreads();
    }
    
    // Write the result to C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    // Allocate host memory
    half *h_A, *h_B;
    float *h_C;
    h_A = (half*)malloc(M * K * sizeof(half));
    h_B = (half*)malloc(K * N * sizeof(half));
    h_C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices with random values between 0 and 1
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(rand() / (float)RAND_MAX);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(rand() / (float)RAND_MAX);
    }
    
    // Allocate device memory
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions for both kernels
    dim3 dimBlock(16, 16);  // 16x16 threads per block
    dim3 dimGrid((N + 15) / 16, (M + 15) / 16);  // Ensure grid covers the entire matrix
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Allocate memory for second result
    float *h_C_shared = (float*)malloc(M * N * sizeof(float));
    
    // Time Tensor Core implementation
    printf("Running Tensor Core implementation...\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    matmul_tensorcore<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float tensorcore_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&tensorcore_time, start, stop));
    printf("Tensor Core time: %.2f ms\n", tensorcore_time);
    
    // Copy tensor core result to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Time Shared Memory implementation
    printf("\nRunning Shared Memory implementation...\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    matmul_shared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float shared_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&shared_time, start, stop));
    printf("Shared Memory time: %.2f ms\n", shared_time);
    
    // Copy shared memory result to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_shared, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\nSpeedup: %.2fx\n", shared_time / tensorcore_time);
    
    // Compare results
    printf("\nComparing results...\n");
    int max_diff_count = 0;
    float max_diff = 0.0f;
    float tolerance = 0.01f; // 2 decimal places tolerance
    
    // Print first 5x5 elements from both implementations
    printf("\nFirst 5x5 elements comparison:\n");
    printf("Row Col | Tensor Core | Shared Mem | Difference\n");
    printf("---------------------------------------------\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            int idx = i * N + j;
            float diff = fabs(h_C[idx] - h_C_shared[idx]);
            printf("%3d %3d | %10.2f | %10.2f | %10.2f\n", 
                   i, j, h_C[idx], h_C_shared[idx], diff);
        }
    }
    
    // Print some random positions
    printf("\nRandom positions comparison:\n");
    printf("Row Col | Tensor Core | Shared Mem | Difference\n");
    printf("---------------------------------------------\n");
    for (int i = 0; i < 5; i++) {
        int row = rand() % M;
        int col = rand() % N;
        int idx = row * N + col;
        float diff = fabs(h_C[idx] - h_C_shared[idx]);
        printf("%3d %3d | %10.2f | %10.2f | %10.2f\n", 
               row, col, h_C[idx], h_C_shared[idx], diff);
    }
    
    // Count differences
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_C[i] - h_C_shared[i]);
        if (diff > tolerance) {
            max_diff_count++;
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    printf("\nSummary:\n");
    if (max_diff_count == 0) {
        printf("Results match within %.2f tolerance\n", tolerance);
    } else {
        printf("Results differ in %d positions (%.2f%%)\n", 
               max_diff_count, (float)max_diff_count / (M * N) * 100);
        printf("Maximum difference: %.2f\n", max_diff);
    }
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_shared);
    
    return 0;
}
