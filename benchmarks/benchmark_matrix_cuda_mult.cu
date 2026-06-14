#include "neural_engine/cuda/CUDA_MatrixMult.hpp"
#include "neural_engine/matrix/Matrix.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

int main() {
    // Benchmark the same matrix sizes used in the CPU matrix multiplication benchmark.
    // This makes the CPU vs CUDA comparison easier to explain in the README.
    std::vector<int> sizes = {128, 256, 512};

    // Run each benchmark multiple times and report the average to reduce timing noise.
    const int trials = 5;

    std::cout << "Matrix object CUDA benchmark:" << std::endl;

    for (int size : sizes) {
        // Create Matrix objects from the engine instead of using raw arrays directly.
        // A and B are randomized once per size so every trial measures the same workload.
        Matrix A = Matrix::matrixRandomization(size, size, 0.0, 1.0);
        Matrix B = Matrix::matrixRandomization(size, size, 0.0, 1.0);
        Matrix C(size, size);

        float totalMilliseconds = 0.0f;

        for (int trial = 0; trial < trials; trial++) {
            cudaEvent_t start;
            cudaEvent_t stop;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // CUDA events measure elapsed GPU time around the matrix multiplication call.
            cudaEventRecord(start);

            // rawData() exposes the Matrix object's contiguous memory to CUDA.
            // The Matrix abstraction stays in the engine, while CUDA receives raw pointers.
            matrixMultiplicationCUDA(
                A.rawData(),
                B.rawData(),
                C.rawData(),
                A.rowsCount(),
                A.colsCount(),
                B.colsCount()
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);

            totalMilliseconds += milliseconds;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        float averageTimeMs = totalMilliseconds / trials;

        std::cout << size << " x " << size
                  << " | Average CUDA time: " << averageTimeMs << " ms"
                  << " | Trials: " << trials
                  << std::endl;
    }

    return 0;
}