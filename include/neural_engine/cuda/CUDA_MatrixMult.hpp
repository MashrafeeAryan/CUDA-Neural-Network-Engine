#pragma once

void matrixMultiplicationCUDA(
    const double* A,
    const double* B,
    double* C,
    int rowsA,
    int colsA,
    int colsB
);

void checkCuda(cudaError_t result, const char* message);