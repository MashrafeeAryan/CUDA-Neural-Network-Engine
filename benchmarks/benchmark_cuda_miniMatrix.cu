
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>



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
}