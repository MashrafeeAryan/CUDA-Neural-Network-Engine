#include "neural_engine/activations/ReLU.hpp"

Matrix ReLU::forward(const Matrix& inputMatrix) const {
    //Create a result matrix with same shape as inputMatrix
    Matrix result(inputMatrix.rowsCount(), inputMatrix.colsCount());

    //This goes through every single row and columna nd says if it is above 0, keep it
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