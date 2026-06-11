/*
 * Dense layer for a feedforward neural network.
 *
 * Forward:  output = input * weights + bias
 * Backward: computes gradients for weights, bias, and input.
 */

#pragma once

#include "neural_engine/matrix/Matrix.hpp"

class Dense {
private:
    int weightInputSize;
    int weightOutputSize;

    Matrix weights;
    Matrix bias;

    // Cached from forward() because backward() needs it for dW = input^T * outputGradient.
    Matrix inputCache;

    Matrix weightGradients;
    Matrix biasGradients;

public:
    Dense(int weightInputSize, int weightOutputSize);

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& outputGradient);

    Matrix getWeightGradients() const;
    Matrix getBiasGradients() const;

    // Applies one SGD update step using gradients from backward().
    void updateParameters(double learningRate);
};