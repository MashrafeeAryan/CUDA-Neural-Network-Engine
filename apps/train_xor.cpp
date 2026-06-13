#include "neural_engine/matrix/Matrix.hpp"
#include "neural_engine/layers/Dense.hpp"
#include "neural_engine/loss/MSE_Loss.hpp"
#include "neural_engine/activations/Sigmoid.hpp"

#include <iostream>

int main() {
    // XOR dataset:
    // The output is 1 when the two inputs are different,
    // and 0 when the two inputs are the same.
    //
    // Input shape:  4 x 2
    // Target shape: 4 x 1
    //
    // Each row is one training example.
    Matrix input(4, 2);

    input(0, 0) = 0.0;
    input(0, 1) = 0.0;

    input(1, 0) = 0.0;
    input(1, 1) = 1.0;

    input(2, 0) = 1.0;
    input(2, 1) = 0.0;

    input(3, 0) = 1.0;
    input(3, 1) = 1.0;

    Matrix target(4, 1);
    target(0, 0) = 0.0;
    target(1, 0) = 1.0;
    target(2, 0) = 1.0;
    target(3, 0) = 0.0;

    // Network architecture:
    // Dense1:        2 input features -> 4 hidden values
    // Hidden sigmoid: adds non-linearity so the model can learn XOR
    // Dense2:        4 hidden values -> 1 output value
    // Output sigmoid: keeps the final prediction between 0 and 1
    Dense dense1(2, 4);
    Sigmoid hiddenSigmoid;

    Dense dense2(4, 1);
    Sigmoid outputSigmoid;

    MSE_Loss mse;

    double learningRate = 1.0;

    for (int epoch = 0; epoch <= 50000; epoch++) {
        // Forward pass:
        // input -> Dense1 -> Sigmoid -> Dense2 -> Sigmoid -> prediction
        Matrix hidden = dense1.forward(input);
        Matrix activatedHidden = hiddenSigmoid.forward(hidden);
        Matrix logits = dense2.forward(activatedHidden);
        Matrix prediction = outputSigmoid.forward(logits);

        double loss = mse.forward(prediction, target);

        // Backward pass:
        // Gradients flow in reverse order through the network.
        Matrix lossGradient = mse.backward(prediction, target);
        Matrix gradientToDense2 = outputSigmoid.backward(lossGradient);
        Matrix gradientToHiddenSigmoid = dense2.backward(gradientToDense2);
        Matrix gradientToDense1 = hiddenSigmoid.backward(gradientToHiddenSigmoid);
        dense1.backward(gradientToDense1);

        // Only Dense layers have trainable parameters.
        // Sigmoid layers transform values and pass gradients, but do not update weights.
        dense1.updateParameters(learningRate);
        dense2.updateParameters(learningRate);

        if (epoch % 5000 == 0) {
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