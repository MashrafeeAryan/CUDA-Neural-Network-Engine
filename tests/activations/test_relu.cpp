#include "neural_engine/activations/ReLU.hpp"

#include <gtest/gtest.h>

TEST(ReLUTest, ForwardTurnsNegativeValuesToZero) {
    ReLU relu;

    Matrix input(2, 2);
    input(0, 0) = -2.0;
    input(0, 1) = 3.0;
    input(1, 0) = 0.0;
    input(1, 1) = -5.0;

    Matrix output = relu.forward(input);

    EXPECT_EQ(output(0, 0), 0.0);
    EXPECT_EQ(output(0, 1), 3.0);
    EXPECT_EQ(output(1, 0), 0.0);
    EXPECT_EQ(output(1, 1), 0.0);
}
TEST(ReLUTest, ForwardKeepsSameShape) {
    ReLU relu;

    Matrix input(3, 4);

    Matrix output = relu.forward(input);

    EXPECT_EQ(output.rowsCount(), 3);
    EXPECT_EQ(output.colsCount(), 4);
}

TEST(ReLUTest, BackwardPassesGradientOnlyForPositiveInputs) {
    ReLU relu;

    Matrix input(2, 2);
    input(0, 0) = -2.0;
    input(0, 1) = 3.0;
    input(1, 0) = 0.0;
    input(1, 1) = 5.0;

    relu.forward(input);

    Matrix outputGradient(2, 2);
    outputGradient(0, 0) = 10.0;
    outputGradient(0, 1) = 20.0;
    outputGradient(1, 0) = 30.0;
    outputGradient(1, 1) = 40.0;

    Matrix inputGradient = relu.backward(outputGradient);

    EXPECT_EQ(inputGradient(0, 0), 0.0);  
    EXPECT_EQ(inputGradient(0, 1), 20.0);  
    EXPECT_EQ(inputGradient(1, 0), 0.0);  
    EXPECT_EQ(inputGradient(1, 1), 40.0); 
}


TEST(ReLUTest, BackwardReturnsSameShapeAsOutputGradient) {
    ReLU relu;

    Matrix input(3, 4);
    relu.forward(input);

    Matrix outputGradient(3, 4);

    Matrix inputGradient = relu.backward(outputGradient);

    EXPECT_EQ(inputGradient.rowsCount(), 3);
    EXPECT_EQ(inputGradient.colsCount(), 4);
}

TEST(ReLUTest, BackwardThrowsForWrongGradientShape) {
    ReLU relu;

    Matrix input(2, 2);
    relu.forward(input);

    Matrix wrongOutputGradient(2, 3);

    EXPECT_THROW(relu.backward(wrongOutputGradient), std::invalid_argument);
}