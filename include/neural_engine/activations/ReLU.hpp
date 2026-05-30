#pragma once

#include "neural_engine/matrix/Matrix.hpp"
class ReLU {
//Handles relu fucntion definitions 
    public:
    Matrix forward(const Matrix& inputMatrix) const;
};