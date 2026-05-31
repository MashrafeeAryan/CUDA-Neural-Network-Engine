#pragma once

#include "neural_engine/matrix/Matrix.hpp"

class MSE_Loss{
    public:
    double forward(const Matrix& predictionMatrix, const Matrix& targetMatrix) const;
    
};