#pragma once

#include "neural_engine/matrix/Matrix.hpp"

class Sigmoid {
    private:
    //We cache sigmoid output because the derivative is y * (1-y)
    //where y is the output from the sigmoid forward pass
    Matrix outputCache;

    public:
    Sigmoid();
    Matrix forward(const Matrix& inputMatrix);
    Matrix backward(const Matrix& outputGradient) const;
};