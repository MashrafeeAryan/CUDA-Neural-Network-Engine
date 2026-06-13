#include "neural_engine/matrix/Matrix.hpp"
#include "neural_engine/layers/Dense.hpp"
#include "neural_engine/loss/MSE_Loss.hpp"

#include <iostream>

int main() {
    // Minimal supervised learning example:
    // train a single Dense layer to learn the mapping 1.0 -> 2.0.
    Matrix input(1, 1);
    input(0, 0) = 1.0;

    Matrix target(1, 1);
    target(0, 0) = 2.0;

    double learningRate = 0.1;

    // One input feature and one output value.
    Dense dense(1, 1);
    MSE_Loss mse;

    for (int epoch = 0; epoch <= 100; epoch++) {
        // Forward pass: compute the model prediction.
        Matrix prediction = dense.forward(input);

        // Measure prediction error using mean squared error.
        double lossValue = mse.forward(prediction, target);

        // Backward pass starts from the loss gradient with respect to prediction.
        Matrix lossGradient = mse.backward(prediction, target);

        // Dense backward computes parameter gradients using the loss gradient.
        dense.backward(lossGradient);

        // Apply one SGD update step to reduce future loss.
        dense.updateParameters(learningRate);

        std::cout << "Epoch " << epoch
                  << " | Prediction: " << prediction(0, 0)
                  << " | Loss: " << lossValue
                  << std::endl;
    }

    return 0;
}