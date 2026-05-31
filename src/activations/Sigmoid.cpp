#include "neural_engine/activations/Sigmoid.hpp"
#include <cmath>
Matrix Sigmoid::forward(const Matrix& inputMatrix) const{

    Matrix result(inputMatrix.rowsCount(), inputMatrix.colsCount());

    for (int r = 0; r < inputMatrix.rowsCount(); r++){
        for (int c=0; c < inputMatrix.colsCount(); c++){
            
            result(r, c) = 1.0/(1.0 + std::exp(-inputMatrix(r, c)));
        }
    }

    return result;

};