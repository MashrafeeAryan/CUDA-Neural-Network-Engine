#include "neural_engine/loss/MSE_Loss.hpp"
#include <stdexcept>
double MSE_Loss::forward(const Matrix& predictionMatrix, const Matrix& targetMatrix) const{
    //The goal of loss functiion is to find the error between both matrices and square it. 
    //Then divide by the number of values in the matrix. 
    //Two ways to do it:
    // 1. We write subtraction functio and just divide by the values
    // 2. We use the subtraction function we wrote for Matrix class and we use the resulting output to calculate the loss.
    // The problem with second way is  we are already doing a loop there and we have to do a loop again to find the squared values
    // SO it is better to just calcuate it again and sqaure it on the fly

    if (predictionMatrix.rowsCount() != targetMatrix.rowsCount() ||
    predictionMatrix.colsCount() != targetMatrix.colsCount()){
        throw std::invalid_argument("Prediction Matrix and Target Matrix shapes don't match");
    }
    double totalSum = 0.0;
    int totalNoValues = predictionMatrix.rowsCount() * predictionMatrix.colsCount();
    for (int r =0; r < predictionMatrix.rowsCount(); r++){
        for (int c=0; c< predictionMatrix.colsCount(); c++){
            // Finds the error and square it by mmultiplying it to itself and add it.
            double difference = ((predictionMatrix(r,c)- targetMatrix(r,c)));
            totalSum+= difference * difference;
        }
    }

    return totalSum/totalNoValues;
}