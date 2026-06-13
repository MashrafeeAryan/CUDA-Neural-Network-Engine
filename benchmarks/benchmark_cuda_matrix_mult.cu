#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matrixMultiplyKernel(
    const double* A,
    const double* B,
    double* C,
    int rowsA,
    int colsA,
    int colsB
) {
    // Map each CUDA thread to one output cell C(row, col).
    // x moves across columns, and y moves down rows.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // The grid is rounded up, so some threads may land outside C.
    // This guard prevents invalid memory access for those extra threads.
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
    // CUDA allocation uses bytes, so convert element counts into byte counts.
    size_t bytesA = rowsA * colsA * sizeof(double);
    size_t bytesB = colsA * colsB * sizeof(double);
    size_t bytesC = rowsA * colsB * sizeof(double);

    // Device pointers store addresses in GPU memory.
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    checkCuda(cudaMalloc(&d_A, bytesA), "Failed to allocate GPU memory for A");
    checkCuda(cudaMalloc(&d_B, bytesB), "Failed to allocate GPU memory for B");
    checkCuda(cudaMalloc(&d_C, bytesC), "Failed to allocate GPU memory for C");

    // Copy input matrices from host memory to device memory.
    checkCuda(cudaMemcpy(d_A, A, bytesA, cudaMemcpyHostToDevice), "Failed to copy A to GPU");
    checkCuda(cudaMemcpy(d_B, B, bytesB, cudaMemcpyHostToDevice), "Failed to copy B to GPU");

    // Each 16 x 16 block computes a 16 x 16 tile of the output matrix.
    dim3 blockSize(16, 16);

    // Round up so the grid covers the full output matrix.
    dim3 gridSize(
        (colsB + blockSize.x - 1) / blockSize.x,
        (rowsA + blockSize.y - 1) / blockSize.y
    );

    matrixMultiplyKernel<<<gridSize, blockSize>>>(
        d_A,
        d_B,
        d_C,
        rowsA,
        colsA,
        colsB
    );

    checkCuda(cudaGetLastError(), "CUDA kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "CUDA kernel execution failed");

    // Copy the computed result back to host memory.
    checkCuda(cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost), "Failed to copy C back to CPU");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    double smallA[4] = {
        1.0, 2.0,
        3.0, 4.0
    };

    double smallB[4] = {
        5.0, 6.0,
        7.0, 8.0
    };

    double smallC[4] = {
        0.0, 0.0,
        0.0, 0.0
    };

    matrixMultiplicationCUDA(
        smallA,
        smallB,
        smallC,
        2,
        2,
        2
    );

    std::cout << "Correctness test result:" << std::endl;
    std::cout << smallC[0] << " " << smallC[1] << std::endl;
    std::cout << smallC[2] << " " << smallC[3] << std::endl;

    std::cout << std::endl;
    std::cout << "CUDA matrix multiplication benchmark:" << std::endl;

    std::vector<int> sizes = {128, 256, 512};

    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int size : sizes) {
        int rowsA = size;
        int colsA = size;
        int colsB = size;

        std::vector<double> A(rowsA * colsA);
        std::vector<double> B(colsA * colsB);
        std::vector<double> C(rowsA * colsB, 0.0);

        // Fill input matrices with deterministic random values.
        // The fixed seed makes benchmark runs easier to reproduce.
        for (double& value : A) {
            value = distribution(generator);
        }

        for (double& value : B) {
            value = distribution(generator);
        }

        cudaEvent_t start;
        cudaEvent_t stop;

        checkCuda(cudaEventCreate(&start), "Failed to create CUDA start event");
        checkCuda(cudaEventCreate(&stop), "Failed to create CUDA stop event");

        checkCuda(cudaEventRecord(start), "Failed to record CUDA start event");

        matrixMultiplicationCUDA(
            A.data(),
            B.data(),
            C.data(),
            rowsA,
            colsA,
            colsB
        );

        checkCuda(cudaEventRecord(stop), "Failed to record CUDA stop event");
        checkCuda(cudaEventSynchronize(stop), "Failed to synchronize CUDA stop event");

        float milliseconds = 0.0f;
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to calculate CUDA elapsed time");

        std::cout << size << " x " << size
                  << " | CUDA time: " << milliseconds << " ms"
                  << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}