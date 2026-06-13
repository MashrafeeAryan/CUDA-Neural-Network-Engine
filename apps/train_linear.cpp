#include "neural_engine/matrix/Matrix.hpp"
#include "neural_engine/layers/Dense.hpp"
#include "neural_engine/loss/MSE_Loss.hpp"

#include <iostream>

int main() {
    // Batch supervised learning example:
    // train a single Dense layer to learn the linear mapping y = 2x.
    Matrix input(4, 1);
    input(0, 0) = 1.0;
    input(1, 0) = 2.0;
    input(2, 0) = 3.0;
    input(3, 0) = 4.0;

    Matrix target(4, 1);
    target(0, 0) = 2.0;
    target(1, 0) = 4.0;
    target(2, 0) = 6.0;
    target(3, 0) = 8.0;

    double learningRate = 0.01;

    // Each row is one training example with one input feature and one target value.
    Dense dense(1, 1);
    MSE_Loss mse;

    for (int epoch = 0; epoch <= 200; epoch++) {
        // Forward pass computes one prediction per input row.
        Matrix prediction = dense.forward(input);

        // MSE measures average error across the full batch.
        double lossValue = mse.forward(prediction, target);

        // Backpropagate the loss signal through the Dense layer.
        Matrix lossGradient = mse.backward(prediction, target);
        dense.backward(lossGradient);

        // Apply one SGD update using gradients accumulated from the batch.
        dense.updateParameters(learningRate);

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Loss: " << lossValue
                      << std::endl;

            std::cout << "Predictions: ";
            for (int r = 0; r < prediction.rowsCount(); r++) {
                std::cout << prediction(r, 0) << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}