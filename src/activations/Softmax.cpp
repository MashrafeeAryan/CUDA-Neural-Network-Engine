#include "neural_engine/activations/Softmax.hpp"
#include <cmath>

//Softmax convverts the the number into probabilities for each row
//Take each element in a row and do e^element
//ADd all of them together
//Divide each element by the sum and we have our sfotmax output
Matrix Softmax::forward(const Matrix& inputMatrix) const {
    Matrix results(inputMatrix.rowsCount(), inputMatrix.colsCount());

    for (int r=0; r<inputMatrix.rowsCount(); r++){
        double totalSumOfRow = 0.0;

        for (int c=0; c<inputMatrix.colsCount(); c++) {
            totalSumOfRow += std::exp(inputMatrix(r, c));
        }

        for (int c=0; c<inputMatrix.colsCount(); c++){
            results(r, c) = std::exp(inputMatrix(r,c))/totalSumOfRow;
        }
    }
    return results;
}