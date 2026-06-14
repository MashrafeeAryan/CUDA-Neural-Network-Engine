#include "neural_engine/matrix/Matrix.hpp"
#include "neural_engine/cuda/CUDA_MatrixMult.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <vector>

int main() {
    // Compares the engine's CPU Matrix multiplication with the CUDA implementation.
    //
    // Matrix multiplication is one of the most expensive operations in dense neural
    // network layers. This benchmark shows the performance difference between the
    // original CPU implementation and the CUDA-accelerated implementation.
    std::vector<int> sizes = {128, 256, 512};
    const int trials = 5;

    std::cout << "CPU vs CUDA Matrix multiplication benchmark:" << std::endl;

    for (int size : sizes) {
        // Use the same randomized Matrix inputs for both CPU and CUDA.
        // This keeps the comparison fair because both implementations multiply
        // the same values with the same matrix dimensions.
        Matrix A = Matrix::matrixRandomization(size, size, 0.0, 1.0);
        Matrix B = Matrix::matrixRandomization(size, size, 0.0, 1.0);

        long long totalCpuTimeMs = 0;

        for (int trial = 0; trial < trials; trial++) {
            // Time only the CPU multiplication step.
            // Matrix creation and randomization happen before timing so the benchmark
            // focuses on multiplication performance rather than setup cost.
            auto start = std::chrono::high_resolution_clock::now();

            Matrix cpuResult = A.matrixMultiplication(B);

            auto end = std::chrono::high_resolution_clock::now();

            auto durationMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            totalCpuTimeMs += durationMs;
        }

        // Average multiple trials to reduce noise from caching, scheduling,
        // and background activity in the runtime environment.
        double averageCpuTimeMs = static_cast<double>(totalCpuTimeMs) / trials;

        Matrix cudaResult(size, size);
        float totalCudaTimeMs = 0.0f;

        for (int trial = 0; trial < trials; trial++) {
            cudaEvent_t start;
            cudaEvent_t stop;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // CUDA events measure elapsed GPU time for the CUDA multiplication call.
            // This is more accurate for GPU work than using a CPU wall-clock timer.
            cudaEventRecord(start);

            // rawData() exposes the Matrix object's contiguous memory to CUDA.
            // The engine keeps the Matrix abstraction, while CUDA receives raw pointers
            // needed for host-to-device copies and kernel execution.
            matrixMultiplicationCUDA(
                A.rawData(),
                B.rawData(),
                cudaResult.rawData(),
                A.rowsCount(),
                A.colsCount(),
                B.colsCount()
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);

            totalCudaTimeMs += milliseconds;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Speedup shows how many times faster CUDA is compared to the CPU baseline.
        float averageCudaTimeMs = totalCudaTimeMs / trials;
        double speedup = averageCpuTimeMs / averageCudaTimeMs;

        std::cout << size << " x " << size
                  << " | Average CPU time: " << averageCpuTimeMs << " ms"
                  << " | Average CUDA time: " << averageCudaTimeMs << " ms"
                  << " | Speedup: " << speedup << "x"
                  << " | Trials: " << trials
                  << std::endl;
    }

    return 0;
}