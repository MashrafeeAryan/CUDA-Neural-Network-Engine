#include "neural_engine/matrix/Matrix.hpp"
#include "neural_engine/layers/Dense.hpp"
#include "neural_engine/activations/ReLU.hpp"

//C++ lubrary that allows program to take input and give output
#include <iostream>
//The goal of it is to prove that eveyrhting is working together:
//Matrix input → Dense layer → ReLU activation → printed output
int main() {
    Matrix input(1,3);
    input(0, 0) = 2.0;
    input(0, 1) = -1.0;
    input(0, 2) = 4.0;


    //Make a dense layer
    //Essentially means 3 input features and 2 output featured
    Dense denseLayer(3,2);

    //Make the activation layer
    ReLU relu;

    //Pass the input to dense layer. aslo known as forward pass
    Matrix denseLayerOutput = denseLayer.forward(input);

    Matrix reluOutput = relu.forward(denseLayerOutput);

    //Prints out the matrix
    for (int r = 0; r< reluOutput.rowsCount(); r++){
        for (int c=0; c< reluOutput.colsCount(); c++){
            std::cout << reluOutput(r,c) << " ";
        }
        std::cout << std::endl;

    }
    return 0;

}