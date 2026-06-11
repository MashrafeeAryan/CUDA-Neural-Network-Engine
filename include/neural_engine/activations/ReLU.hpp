#pragma once

#include "neural_engine/matrix/Matrix.hpp"
class ReLU {
//Handles relu fucntion definitions 
    private:    
    Matrix inputCache;

    public:
    ReLU();
    Matrix forward(const Matrix& inputMatrix);
    Matrix backward(const Matrix& outputGradient);
};