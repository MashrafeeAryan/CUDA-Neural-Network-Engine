#include "neural_engine/layers/Dense.hpp"

#include <stdexcept>

Dense::Dense(int weightInputSize, int weightOutputSize)
    : weightInputSize(weightInputSize),
      weightOutputSize(weightOutputSize),
      weights(Matrix::matrixRandomization(weightInputSize, weightOutputSize, -0.1, 0.1)),
      inputCache(1, weightInputSize),
      weightGradients(weightInputSize, weightOutputSize),
      biasGradients(1, weightOutputSize),
      bias(1, weightOutputSize) {
}

Matrix Dense::forward(const Matrix& input) {
    if (input.colsCount() != weightInputSize) {
        throw std::invalid_argument(
            "Input columns must match Dense layer input size"
        );
    }

    // Cache the input because Dense backward needs it for:
    // dW = input^T * outputGradient
    inputCache = input;

    // Dense forward pass:
    // output = input * weights + bias
    Matrix output = input.matrixMultiplication(weights);
    output = output.addRowVector(bias);

    return output;
}

Matrix Dense::backward(const Matrix& outputGradient) {
    // outputGradient must match the Dense output shape:
    // batchSize x weightOutputSize
    if (outputGradient.rowsCount() != inputCache.rowsCount() ||
        outputGradient.colsCount() != weightOutputSize) {
        throw std::invalid_argument(
            "Output gradient shape does not match Dense output shape"
        );
    }

    // Weight gradients:
    // dW = input^T * outputGradient
    weightGradients =
        inputCache.transpose().matrixMultiplication(outputGradient);

    // Bias is shared across the batch, so its gradient is the
    // column-wise sum of output gradients.
    biasGradients = Matrix(1, outputGradient.colsCount());

    for (int r = 0; r < outputGradient.rowsCount(); r++) {
        for (int c = 0; c < outputGradient.colsCount(); c++) {
            biasGradients(0, c) += outputGradient(r, c);
        }
    }

    // Input gradient is passed to the previous layer:
    // dInput = outputGradient * weights^T
    Matrix inputGradient =
        outputGradient.matrixMultiplication(weights.transpose());

    return inputGradient;
}

Matrix Dense::getWeightGradients() const {
    return weightGradients;
}

Matrix Dense::getBiasGradients() const {
    return biasGradients;
}

void Dense::updateParameters(double learningRate) {
    // Apply one SGD update step using gradients computed during backward().
    // parameter = parameter - learningRate * gradient
    for (int r = 0; r < weights.rowsCount(); r++) {
        for (int c = 0; c < weights.colsCount(); c++) {
            weights(r, c) =
                weights(r, c) - learningRate * weightGradients(r, c);
        }
    }

    for (int r = 0; r < bias.rowsCount(); r++) {
        for (int c = 0; c < bias.colsCount(); c++) {
            bias(r, c) =
                bias(r, c) - learningRate * biasGradients(r, c);
        }
    }
}