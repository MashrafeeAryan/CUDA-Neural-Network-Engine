#include "neural_engine/matrix/Matrix.hpp"

#include <chrono>
#include <iostream>
#include <vector>

int main() {
    // CPU matrix multiplication benchmark.
    //
    // Dense layers rely heavily on matrix multiplication during forward
    // and backward passes, so this benchmark establishes a CPU baseline
    // before adding CUDA acceleration.
    std::vector<int> sizes = {128, 256, 512};
    int trials = 5;

    for (int size : sizes) {
        long long totalTimeMs = 0;

        for (int trialNo = 1; trialNo <= trials; trialNo++) {
            Matrix A = Matrix::matrixRandomization(size, size, -1.0, 1.0);
            Matrix B = Matrix::matrixRandomization(size, size, -1.0, 1.0);

            // Time only the matrix multiplication step.
            // Matrix creation/randomization is intentionally outside the timed region.
            auto start = std::chrono::high_resolution_clock::now();

            Matrix C = A.matrixMultiplication(B);

            auto end = std::chrono::high_resolution_clock::now();

            auto durationMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            totalTimeMs += durationMs;
        }

        // Average multiple trials to reduce noise from cache effects,
        // OS scheduling, and background system activity.
        double averageTimeMs = static_cast<double>(totalTimeMs) / trials;

        std::cout << "Matrix size: " << size << " x " << size
                  << " | Average CPU time: " << averageTimeMs << " ms"
                  << " | Trials: " << trials
                  << std::endl;
    }

    return 0;
}