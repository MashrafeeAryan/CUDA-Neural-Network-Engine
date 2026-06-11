#include "neural_engine/activations/ReLU.hpp"
#include <stdexcept>


ReLU::ReLU()
    : inputCache(1,1){}



Matrix ReLU::forward(const Matrix& inputMatrix)  {
    // Create a result matrix with same shape as inputMatrix
    Matrix result(inputMatrix.rowsCount(), inputMatrix.colsCount());
    inputCache = inputMatrix;
    // This goes through every single row and columna nd says if it is above 0, keep it
    // if it is below 0 make it 0
    for (int r = 0; r <inputMatrix.rowsCount(); r++ ){
        for (int c=0; c<inputMatrix.colsCount(); c++){
            if (inputMatrix(r,c) >= 0) {
                result(r,c) = (inputMatrix)(r,c);
            }
            else{
                result(r,c) = 0;
            }
        }
    }
    return result;
}

// ReLU passes gradients only for values that were positive during the forward pass.
// Values that were zero or negative are blocked because ReLU output was flat there.
Matrix ReLU::backward(const Matrix& outputGradient){
    if (outputGradient.rowsCount() != inputCache.rowsCount() ||
        outputGradient.colsCount() != inputCache.colsCount()) {
        throw std::invalid_argument(
            "Output gradient shape must match cached ReLU input shape"
        );
    }
    Matrix result(outputGradient.rowsCount(), outputGradient.colsCount());
    for (int r=0; r <outputGradient.rowsCount(); r++){
        for (int c=0; c<outputGradient.colsCount(); c++){
            if (inputCache(r, c) <= 0){
                result(r,c) = 0;
            }
            else{
                result(r,c) = outputGradient(r,c);
            }

        }
    }
    return result;
}