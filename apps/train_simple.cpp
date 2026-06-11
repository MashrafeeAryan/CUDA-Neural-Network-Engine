#include "neural_engine/matrix/Matrix.hpp"
#include "neural_engine/layers/Dense.hpp"
#include "neural_engine/loss/MSE_Loss.hpp"
#include <iostream>
int main() {

    Matrix input(1,1);
    input(0,0) = 1.0;

    Matrix target(1,1);
    target(0,0) = 2.0;
    
    double learningRate = 0.1;
    Dense dense(1,1);
    MSE_Loss mse;

    for (int epoch =0; epoch<=100; epoch++){
   

        Matrix prediction = dense.forward(input);

        double lossValue = mse.forward(prediction, target);

        Matrix lossGradient = mse.backward(prediction, target);

        dense.backward(lossGradient);

        dense.updateParameters(learningRate);

        std::cout << "Epoch " << epoch
                      << " | Prediction: " << prediction(0, 0)
                      << " | Loss: " << lossValue
                      << std::endl;
    }
}