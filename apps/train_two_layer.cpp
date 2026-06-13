#include "neural_engine/matrix/Matrix.hpp"
#include "neural_engine/layers/Dense.hpp"
#include "neural_engine/loss/MSE_Loss.hpp"
#include "neural_engine/activations/Sigmoid.hpp"

#include <iostream>

int main() {
    // Batch training example:
    // Learn the linear mapping y = 2x using a two-layer neural network.
    //
    // Input shape:  4 x 1
    // Target shape: 4 x 1
    
    // Each row is one training example.
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

    // Network architecture:
    // Dense1:  1 input feature  -> 2 hidden values
    // Sigmoid: applies non-linearity element-wise
    // Dense2:  2 hidden values -> 1 output prediction
    Dense dense1(1, 2);
    Sigmoid sigmoid;
    Dense dense2(2, 1);

    MSE_Loss mse;

    for (int epoch = 0; epoch <= 1000; epoch++) {
        // Forward pass:
        // input -> Dense1 -> Sigmoid -> Dense2 -> prediction
        Matrix hidden = dense1.forward(input);
        Matrix activatedHidden = sigmoid.forward(hidden);
        Matrix prediction = dense2.forward(activatedHidden);

        // MSE measures the average error across the full batch.
        double loss = mse.forward(prediction, target);

        // Backward pass:
        // Gradients flow in reverse order through the network.
        Matrix lossGradient = mse.backward(prediction, target);
        Matrix gradientToSigmoid = dense2.backward(lossGradient);
        Matrix gradientToDense1 = sigmoid.backward(gradientToSigmoid);
        dense1.backward(gradientToDense1);

        // Update trainable parameters after all gradients are computed.
        // Sigmoid has no trainable parameters, so only Dense layers update.
        dense1.updateParameters(learningRate);
        dense2.updateParameters(learningRate);

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Loss: " << loss
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