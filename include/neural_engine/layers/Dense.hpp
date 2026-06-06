/*
Dense Layer takes input matrix, mutliplies it by a weights amtrix, adds a bias, and produces outut
output = input * weights + bias
*/
//Preprocessor directive to esnure header file is included only once during compiling
#pragma once
 
//Since we are using matrix object, we need:
#include "neural_engine/matrix/Matrix.hpp"

#include <vector>
class Dense{
    private:
    //how many features each input has
    int weightInputSize;
    //how many neuirons/outputs this layer of ours are creating
    int weightOutputSize;
    
    Matrix weights;
    Matrix bias;

    // Saves the input for backward pass
    Matrix inputCache;
    Matrix weightGradients;
    Matrix biasGradients;
    public:
    //Constructor creates a layer with the input and output size
    Dense(int weightInputSize, int weightOutputSize);

    //Forward function is responsible to pass the data through the layer
    //It takes the input matrix, multiply it with the layer's weights, add bias
    //then return the output
    Matrix forward(const Matrix& input);

    //Dense backward function uses the ouitput gradietn to find, weights and bias gradient to change weights and bias
    Matrix backward(const Matrix& outputGradient);

    Matrix getWeightGradient() const;
    Matrix getBiasGradient() const;
    Matrix getInputGradient() const;

};