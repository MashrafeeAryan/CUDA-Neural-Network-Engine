#include "neural_engine/activations/Sigmoid.hpp"

#include <cmath>
#include <stdexcept>

Sigmoid::Sigmoid()
    : outputCache(1, 1) {
}

Matrix Sigmoid::forward(const Matrix& inputMatrix) {
    Matrix result(inputMatrix.rowsCount(), inputMatrix.colsCount());

    for (int r = 0; r < inputMatrix.rowsCount(); r++) {
        for (int c = 0; c < inputMatrix.colsCount(); c++) {
            result(r, c) = 1.0 / (1.0 + std::exp(-inputMatrix(r, c)));
        }
    }

    // Cache the sigmoid output because backward uses y * (1 - y),
    // where y is the output from the forward pass.
    outputCache = result;

    return result;
}

// Sigmoid backward is element-wise:
// dL/dx = dL/dy * y * (1 - y)
//
// outputGradient represents dL/dy.
// outputCache stores y from the forward pass.
// The returned matrix is dL/dx, the gradient passed to the previous layer.
Matrix Sigmoid::backward(const Matrix& outputGradient) const {
    if (outputGradient.rowsCount() != outputCache.rowsCount() ||
        outputGradient.colsCount() != outputCache.colsCount()) {
        throw std::invalid_argument(
            "Output gradient shape must match cached Sigmoid output shape"
        );
    }

    Matrix inputGradient(outputGradient.rowsCount(), outputGradient.colsCount());

    for (int r = 0; r < outputGradient.rowsCount(); r++) {
        for (int c = 0; c < outputGradient.colsCount(); c++) {
            inputGradient(r, c) =
                outputGradient(r, c) * outputCache(r, c) * (1.0 - outputCache(r, c));
        }
    }

    return inputGradient;
}