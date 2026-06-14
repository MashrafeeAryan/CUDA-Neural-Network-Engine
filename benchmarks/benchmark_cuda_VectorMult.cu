#include "neural_engine/cuda/CUDA_MatrixMult.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>


int main() {
    std::vector<int> sizes = {128, 256, 512};

    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const int trials = 5;

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

      
        float totalMilliseconds  = 0.0f;
        for (int trial=0; trial<trials;trial++){
            cudaEvent_t start;
            cudaEvent_t stop;
            std::fill(C.begin(), C.end(), 0.0);
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            matrixMultiplicationCUDA(
                A.data(),
                B.data(),
                C.data(),
                rowsA,
                colsA,
                colsB
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);
            totalMilliseconds+=milliseconds;
    

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }  
        float averageTimeMs= totalMilliseconds/trials; 
        std::cout << size << " x " << size
                  << " | Average CUDA time: " << averageTimeMs << " ms"
                  << " | Trials: " << trials 
                  << std::endl;
    }

    return 0;
}