#include "neural_engine/layers/Dense.hpp"
#include <random>
#include <stdexcept>


//Since input is a matrix object and colsCount() is a public function of the  matrix class we can call any function on th einput object
Matrix Dense::forward(const Matrix& input){
    //inputSize means number of columns the input matrix must have
    if (input.colsCount() != weightInputSize){
        throw std::invalid_argument("Matrix dimensions must match for multiplication. Number of columsn of first amtrix should equal number of rows of second matrix");
    }
    //Save the input in input cache for backward pass
    inputCache = input;

    //Use the input matrix and mutliply with weihgts matrix
    // Then add the bias to the output matrix
    Matrix output = input.matrixMultiplication(weights);
    output = output.addRowVector(bias);


    return output;
}

// Constructor with member's intialization
//Constructor makes oject ready to use immedietly.
//The variables you are passing to a class should do something 
//And cosntrutor helps does that as soon as you create an object of that class
Dense::Dense(int weightInputSize, int weightOutputSize):
    weightInputSize(weightInputSize),
    weightOutputSize(weightOutputSize),
    //Set the weights variable to have a matrix of random variables
    weights(Matrix::matrixRandomization(weightInputSize, weightOutputSize, -0.1, 0.1)),
    inputCache(1, weightInputSize),
    weightGradients(weightInputSize, weightOutputSize),
    biasGradients(1, weightOutputSize),
    
    //Set the bias
    //Bias is 1 row and weightOutputSiz columns
    bias(1, weightOutputSize){

    }

Matrix Dense::backward(const Matrix& outputGradient){
    // Formulas
    // weightGradients = inputCacheᵀ × outputGradient
    // biasGradients   = sum outputGradient down each column
    // inputGradient   = outputGradient × weightsᵀ  
    //T means transpose allows us to multiply
    if (outputGradient.rowsCount() != inputCache.rowsCount() || outputGradient.colsCount() != weightOutputSize){
        throw std::invalid_argument(
            "Output gradeitn shape does not match Dense output shape"
        );
    }
    Matrix transposedInputCache = inputCache.transpose();
    weightGradients = transposedInputCache.matrixMultiplication(outputGradient);

    //rest biasGradients to -
    biasGradients = Matrix(1, outputGradient.colsCount());
    
    //Find bias gradient
    for (int r =0; r < outputGradient.rowsCount(); r++){
        for (int c =0; c< outputGradient.colsCount(); c++){
            biasGradients(0, c) += outputGradient(r,c);
        }
    }

    Matrix transposedWeights = weights.transpose();
    Matrix inputGradients = outputGradient.matrixMultiplication(transposedWeights);

    return inputGradients;
}

Matrix Dense::getWeightGradients() const{
    return weightGradients;
}

Matrix Dense::getBiasGradients() const {
    return biasGradients;
}