#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMultiplyKernel(
    const double* A,
    const double* B,
    double* C,
    int rowsA,
    int colsA,
    int colsB
) {
    // Map each CUDA thread to one output cell C(row, col).
    // x represents columns and y represents rows in this 2D grid.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Some launched threads may fall outside the matrix when dimensions
    // are not divisible by the block size, so guard against out-of-bounds access.
    if (row < rowsA && col < colsB) {
        double sum = 0.0;

        // Compute the dot product between row `row` of A and column `col` of B.
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }

        C[row * colsB + col] = sum;
    }
}

void matrixMultiplicationCUDA(
    const double* A,
    const double* B,
    double* C,
    int rowsA,
    int colsA,
    int colsB
) {
    // Matrix shapes:
    // A = rowsA x colsA
    // B = colsA x colsB
    // C = rowsA x colsB
    //
    // CUDA memory allocation uses byte counts, so convert the number
    // of double values in each matrix into bytes.
    size_t bytesA = rowsA * colsA * sizeof(double);
    size_t bytesB = colsA * colsB * sizeof(double);
    size_t bytesC = rowsA * colsB * sizeof(double);

    // Device pointers store addresses in GPU memory.
    // Host = CPU memory, Device = GPU memory.
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    // Allocate GPU memory for input matrices A and B, and output matrix C.
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    // Copy input data from CPU memory to GPU memory.
    cudaMemcpy(d_A, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytesB, cudaMemcpyHostToDevice);

    // Use a 16 x 16 thread block. Each block computes a 16 x 16 tile of C.
    dim3 blockSize(16, 16);

    // Round up so the grid covers the full output matrix, even when
    // rowsA or colsB are not divisible by the block dimensions.
    dim3 gridSize(
        (colsB + blockSize.x - 1) / blockSize.x,
        (rowsA + blockSize.y - 1) / blockSize.y
    );

    // Launch the CUDA kernel.
    // gridSize = number of blocks
    // blockSize = number of threads per block
    matrixMultiplyKernel<<<gridSize, blockSize>>>(
        d_A,
        d_B,
        d_C,
        rowsA,
        colsA,
        colsB
    );

    // Wait for the GPU to finish before copying the result back.
    cudaDeviceSynchronize();

    // Copy the computed result from GPU memory back to CPU memory.
    cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost);

    // Release GPU memory after the computation is complete.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    double A[4] = {
        1.0, 2.0,
        3.0, 4.0
    };

    double B[4] = {
        5.0, 6.0,
        7.0, 8.0
    };

    double C[4] = {
        0.0, 0.0,
        0.0, 0.0
    };

    matrixMultiplicationCUDA(
        A,
        B,
        C,
        2,
        2,
        2
    );

    std::cout << "Result C:" << std::endl;
    std::cout << C[0] << " " << C[1] << std::endl;
    std::cout << C[2] << " " << C[3] << std::endl;

    return 0;
}