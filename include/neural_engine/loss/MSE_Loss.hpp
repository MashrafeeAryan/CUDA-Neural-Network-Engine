#pragma once

#include "neural_engine/matrix/Matrix.hpp"

class MSE_Loss{
    public:
    double forward(const Matrix& predictionMatrix, const Matrix& targetMatrix) const;
    
    //Backward pass figures out how does the loss change if prediction changes
    // It is also know an as gradient
    //For MSE the gradient is:
    // 2 * (prediction - target) / number_of_values
    //Negative gradient -> value should increase
    //positive gradient -> value should decrease
    // zero gradient -> value is already correct
    Matrix backward(const Matrix& predictionMatrix, const Matrix& targetMatrix) const;

};


